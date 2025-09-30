"""
Offline recommendation engine to replace cloud-based recommendations.
This module provides content-based and collaborative filtering algorithms
to generate music recommendations without external API calls.
"""

from collections import defaultdict, Counter
from typing import List, Dict, Set, Tuple, Optional
import random
import math
from dataclasses import dataclass
from rapidfuzz import fuzz

from swingmusic.models.track import Track
from swingmusic.models.album import Album
from swingmusic.models.artist import Artist
from swingmusic.store.tracks import TrackStore
from swingmusic.store.albums import AlbumStore
from swingmusic.store.artists import ArtistStore
from swingmusic.db.userdata import SimilarArtistTable
from swingmusic.utils.hashing import create_hash


@dataclass
class SimilarityScore:
    """Represents a similarity score between two items."""
    item_hash: str
    score: float
    reasons: List[str] = None

    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []


class OfflineRecommendationEngine:
    """
    Offline recommendation engine that provides music recommendations
    based on content similarity and collaborative filtering.
    """

    def __init__(self):
        self.track_store = TrackStore
        self.album_store = AlbumStore
        self.artist_store = ArtistStore

        # Cache for performance
        self._track_similarity_cache: Dict[str, List[SimilarityScore]] = {}
        self._artist_similarity_cache: Dict[str, List[str]] = {}

    def get_similar_tracks(self, seed_tracks: List[Track], limit: int = 50) -> List[str]:
        """
        Get similar tracks based on seed tracks.

        Args:
            seed_tracks: List of seed tracks to base recommendations on
            limit: Maximum number of similar tracks to return

        Returns:
            List of weakhashes of similar tracks
        """
        if not seed_tracks:
            return []

        # Calculate similarity scores for all tracks
        all_scores: Dict[str, float] = defaultdict(float)
        track_weights: Dict[str, int] = {}

        for seed_track in seed_tracks:
            similar_tracks = self._get_similar_tracks_for_track(seed_track, limit=100)

            for score in similar_tracks:
                # Weight by how many seed tracks this track is similar to
                all_scores[score.item_hash] += score.score
                track_weights[score.item_hash] = track_weights.get(score.item_hash, 0) + 1

        # Boost scores for tracks similar to multiple seed tracks
        for track_hash in all_scores:
            if track_weights[track_hash] > 1:
                all_scores[track_hash] *= (1 + 0.1 * track_weights[track_hash])

        # Sort by score and return top results
        sorted_tracks = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)

        # Filter out seed tracks and return weakhashes
        seed_weakhashes = {t.weakhash for t in seed_tracks}
        result = []

        for track_hash, score in sorted_tracks[:limit]:
            track = self.track_store.trackhashmap.get(track_hash)
            if track and track.weakhash not in seed_weakhashes:
                result.append(track.weakhash)

        return result

    def get_similar_albums(self, seed_tracks: List[Track], limit: int = 20) -> List[str]:
        """
        Get similar albums based on seed tracks.

        Args:
            seed_tracks: List of seed tracks to base recommendations on
            limit: Maximum number of similar albums to return

        Returns:
            List of album weakhashes
        """
        if not seed_tracks:
            return []

        # Extract unique album weakhashes from seed tracks
        seed_album_weakhashes = set()
        for track in seed_tracks:
            album = self.album_store.albummap.get(track.albumhash)
            if album:
                seed_album_weakhashes.add(album.album.weakhash)

        # Find similar albums based on artists and genres
        all_scores: Dict[str, float] = defaultdict(float)

        for seed_track in seed_tracks:
            similar_albums = self._get_similar_albums_for_track(seed_track, limit=50)

            for score in similar_albums:
                all_scores[score.item_hash] += score.score

        # Sort by score
        sorted_albums = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)

        # Filter out seed albums and return weakhashes
        result = []
        for album_weakhash, score in sorted_albums[:limit]:
            if album_weakhash not in seed_album_weakhashes:
                result.append(album_weakhash)

        return result

    def get_similar_artists(self, seed_tracks: List[Track], limit: int = 20) -> List[str]:
        """
        Get similar artists based on seed tracks.

        Args:
            seed_tracks: List of seed tracks to base recommendations on
            limit: Maximum number of similar artists to return

        Returns:
            List of artist hashes
        """
        if not seed_tracks:
            return []

        # Extract unique artist hashes from seed tracks
        seed_artist_hashes = set()
        for track in seed_tracks:
            for artisthash in track.artisthashes:
                seed_artist_hashes.add(artisthash)

        # Get similar artists using stored LastFM data and content similarity
        all_scores: Dict[str, float] = defaultdict(float)

        for seed_track in seed_tracks:
            for artisthash in seed_track.artisthashes:
                similar_artists = self._get_similar_artists_for_artist(artisthash, limit=50)

                for similar_hash in similar_artists:
                    all_scores[similar_hash] += 1.0  # Simple counting for now

        # Sort by frequency
        sorted_artists = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)

        # Filter out seed artists
        result = []
        for artist_hash, score in sorted_artists[:limit]:
            if artist_hash not in seed_artist_hashes:
                result.append(artist_hash)

        return result

    def _get_similar_tracks_for_track(self, track: Track, limit: int = 50) -> List[SimilarityScore]:
        """
        Get similar tracks for a single track based on content similarity.
        """
        cache_key = track.trackhash
        if cache_key in self._track_similarity_cache:
            return self._track_similarity_cache[cache_key]

        scores: Dict[str, SimilarityScore] = {}

        # Get all tracks for comparison
        all_tracks = self.track_store.get_flat_list()

        for other_track in all_tracks:
            if other_track.trackhash == track.trackhash:
                continue

            score = self._calculate_track_similarity(track, other_track)
            if score.score > 0:
                scores[other_track.trackhash] = score

        # Sort and cache results
        sorted_scores = sorted(scores.values(), key=lambda x: x.score, reverse=True)[:limit]
        self._track_similarity_cache[cache_key] = sorted_scores

        return sorted_scores

    def _calculate_track_similarity(self, track1: Track, track2: Track) -> SimilarityScore:
        """
        Calculate similarity score between two tracks.
        """
        score = 0.0
        reasons = []

        # Artist similarity (most important)
        artist_similarity = self._calculate_artist_overlap(track1, track2)
        if artist_similarity > 0:
            score += artist_similarity * 3.0
            reasons.append(f"artist_overlap:{artist_similarity}")

        # Genre similarity
        genre_similarity = self._calculate_genre_overlap(track1, track2)
        if genre_similarity > 0:
            score += genre_similarity * 2.0
            reasons.append(f"genre_overlap:{genre_similarity}")

        # Album similarity (if different albums by same artist)
        if (track1.albumhash != track2.albumhash and
            any(a in track2.artisthashes for a in track1.artisthashes)):
            score += 1.0
            reasons.append("same_artist_different_album")

        # Title similarity using fuzzy matching (lower weight)
        title_similarity = fuzz.ratio(track1.title.lower(), track2.title.lower()) / 100.0
        if title_similarity > 0.6:  # Only count significant similarities
            score += title_similarity * 0.5
            reasons.append(f"title_similarity:{title_similarity:.2f}")

        return SimilarityScore(
            item_hash=track2.trackhash,
            score=score,
            reasons=reasons
        )

    def _calculate_artist_overlap(self, track1: Track, track2: Track) -> float:
        """Calculate artist overlap between two tracks."""
        track1_artists = set(track1.artisthashes)
        track2_artists = set(track2.artisthashes)

        if not track1_artists or not track2_artists:
            return 0.0

        intersection = track1_artists.intersection(track2_artists)
        union = track1_artists.union(track2_artists)

        # Jaccard similarity
        return len(intersection) / len(union) if union else 0.0

    def _calculate_genre_overlap(self, track1: Track, track2: Track) -> float:
        """Calculate genre overlap between two tracks."""
        track1_genres = set(track1.genrehashes) if track1.genrehashes else set()
        track2_genres = set(track2.genrehashes) if track2.genrehashes else set()

        if not track1_genres or not track2_genres:
            return 0.0

        intersection = track1_genres.intersection(track2_genres)
        union = track1_genres.union(track2_genres)

        # Jaccard similarity
        return len(intersection) / len(union) if union else 0.0

    def _get_similar_albums_for_track(self, track: Track, limit: int = 20) -> List[SimilarityScore]:
        """Get similar albums for a track."""
        scores: Dict[str, SimilarityScore] = {}

        album = self.album_store.albummap.get(track.albumhash)
        if not album:
            return []

        # Find albums by same artists
        for artisthash in track.artisthashes:
            artist = self.artist_store.artistmap.get(artisthash)
            if artist:
                for albumhash in artist.albumhashes:
                    other_album = self.album_store.albummap.get(albumhash)
                    if other_album and other_album.album.albumhash != track.albumhash:
                        score = self._calculate_album_similarity(album.album, other_album.album)
                        if score.score > 0:
                            scores[other_album.album.weakhash] = score

        # Sort and return top results
        sorted_scores = sorted(scores.values(), key=lambda x: x.score, reverse=True)[:limit]
        return sorted_scores

    def _calculate_album_similarity(self, album1: Album, album2: Album) -> SimilarityScore:
        """Calculate similarity between two albums."""
        score = 0.0
        reasons = []

        # Artist similarity
        album1_artists = set(album1.artisthashes)
        album2_artists = set(album2.artisthashes)

        artist_overlap = len(album1_artists.intersection(album2_artists)) / len(album1_artists.union(album2_artists))
        if artist_overlap > 0:
            score += artist_overlap * 3.0
            reasons.append(f"artist_overlap:{artist_overlap:.2f}")

        # Genre similarity
        album1_genres = set(album1.genrehashes) if album1.genrehashes else set()
        album2_genres = set(album2.genrehashes) if album2.genrehashes else set()

        genre_overlap = len(album1_genres.intersection(album2_genres)) / len(album1_genres.union(album2_genres)) if album1_genres or album2_genres else 0
        if genre_overlap > 0:
            score += genre_overlap * 2.0
            reasons.append(f"genre_overlap:{genre_overlap:.2f}")

        # Title similarity (for series/compilations)
        title_similarity = fuzz.ratio(album1.title.lower(), album2.title.lower()) / 100.0
        if title_similarity > 0.7:
            score += title_similarity * 1.0
            reasons.append(f"title_similarity:{title_similarity:.2f}")

        return SimilarityScore(
            item_hash=album2.weakhash,
            score=score,
            reasons=reasons
        )

    def _get_similar_artists_for_artist(self, artisthash: str, limit: int = 20) -> List[str]:
        """
        Get similar artists for an artist using stored LastFM data.
        """
        if artisthash in self._artist_similarity_cache:
            return self._artist_similarity_cache[artisthash]

        # Try to get from database first
        similar_artist_data = SimilarArtistTable.get_by_hash(artisthash)
        if similar_artist_data:
            # Convert to list of hashes and cache
            similar_hashes = list(similar_artist_data.similar_artists.keys())
            self._artist_similarity_cache[artisthash] = similar_hashes[:limit]
            return similar_hashes[:limit]

        # Fallback: find artists with similar genres
        artist = self.artist_store.artistmap.get(artisthash)
        if not artist:
            return []

        similar_hashes = []
        target_genres = set(artist.artist.genrehashes) if artist.artist.genrehashes else set()

        for other_artist_hash, other_artist in self.artist_store.artistmap.items():
            if other_artist_hash == artisthash:
                continue

            other_genres = set(other_artist.artist.genrehashes) if other_artist.artist.genrehashes else set()
            if target_genres and other_genres:
                overlap = len(target_genres.intersection(other_genres)) / len(target_genres.union(other_genres))
                if overlap > 0.3:  # At least 30% genre overlap
                    similar_hashes.append(other_artist_hash)

        # Cache and return
        self._artist_similarity_cache[artisthash] = similar_hashes[:limit]
        return similar_hashes[:limit]


# Global instance for use throughout the application
offline_engine = OfflineRecommendationEngine()
