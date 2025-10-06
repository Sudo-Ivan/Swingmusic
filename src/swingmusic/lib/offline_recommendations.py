"""
Offline recommendation engine to replace cloud-based recommendations.
This module provides content-based and collaborative filtering algorithms
to generate music recommendations without external API calls.
"""

from collections import defaultdict, Counter, OrderedDict
from typing import List, Dict, Set, Tuple, Optional, Any
import random
import math
import time
from dataclasses import dataclass, field
from rapidfuzz import fuzz
import heapq

from swingmusic.models.track import Track
from swingmusic.models.album import Album
from swingmusic.models.artist import Artist
from swingmusic.store.tracks import TrackStore
from swingmusic.store.albums import AlbumStore
from swingmusic.store.artists import ArtistStore
from swingmusic.db.userdata import SimilarArtistTable, ScrobbleTable
from swingmusic.utils.hashing import create_hash


@dataclass
class SimilarityScore:
    """Represents a similarity score between two items."""
    item_hash: str
    score: float
    reasons: List[str] = field(default_factory=list)
    confidence: float = 0.0
    diversity_score: float = 0.0
    temporal_boost: float = 0.0

    def final_score(self, weights: Dict[str, float] = None) -> float:
        """Calculate final weighted score."""
        if weights is None:
            weights = {'similarity': 1.0, 'diversity': 0.1, 'temporal': 0.05}

        return (
            self.score * weights['similarity'] +
            self.diversity_score * weights['diversity'] +
            self.temporal_boost * weights['temporal']
        )

    def __lt__(self, other):
        """Less than comparison for heap operations."""
        if not isinstance(other, SimilarityScore):
            return NotImplemented
        return self.final_score() < other.final_score()

    def __le__(self, other):
        """Less than or equal comparison."""
        if not isinstance(other, SimilarityScore):
            return NotImplemented
        return self.final_score() <= other.final_score()

    def __gt__(self, other):
        """Greater than comparison."""
        if not isinstance(other, SimilarityScore):
            return NotImplemented
        return self.final_score() > other.final_score()

    def __ge__(self, other):
        """Greater than or equal comparison."""
        if not isinstance(other, SimilarityScore):
            return NotImplemented
        return self.final_score() >= other.final_score()


@dataclass
class RecommendationResult:
    """Complete recommendation result with metadata."""
    item_hash: str
    score: float
    reasons: List[str]
    confidence: float
    rank: int = 0
    item_type: str = "track"  # track, album, artist


class OfflineRecommendationEngine:
    """
    Offline recommendation engine that provides music recommendations
    based on content similarity and collaborative filtering.
    """

    def __init__(self):
        self.track_store = TrackStore
        self.album_store = AlbumStore
        self.artist_store = ArtistStore

        # Enhanced caching with TTL and LRU
        self._track_similarity_cache: OrderedDict[str, Tuple[List[SimilarityScore], float]] = OrderedDict()
        self._artist_similarity_cache: OrderedDict[str, Tuple[List[str], float]] = OrderedDict()
        self._user_profile_cache: Dict[int, Dict[str, Any]] = {}

        # Cache configuration
        self.CACHE_TTL = 3600  # 1 hour
        self.MAX_CACHE_SIZE = 1000

        # Pre-computed user profiles for collaborative filtering
        self._user_item_matrix: Dict[int, Dict[str, float]] = {}
        self._item_user_matrix: Dict[str, Dict[int, float]] = {}

        # Diversity tracking
        self._recent_recommendations: Dict[str, Set[str]] = defaultdict(set)

        # Performance monitoring
        self._cache_hits = 0
        self._cache_requests = 0

    def _extract_weakhashes(self, tracks_or_groups: List[Any]) -> Set[str]:
        """
        Safely extract weakhashes from a list that may contain Track or TrackGroup objects.
        """
        weakhashes = set()
        for item in tracks_or_groups:
            # Check if it's a TrackGroup (has tracks attribute)
            if hasattr(item, 'tracks') and hasattr(item, 'get_best'):
                # It's a TrackGroup, get weakhash from best track
                best_track = item.get_best()
                weakhashes.add(best_track.weakhash)
            elif hasattr(item, 'weakhash'):
                # It's a Track object
                weakhashes.add(item.weakhash)
        return weakhashes

    def _extract_tracks(self, tracks_or_groups: List[Any]) -> List[Track]:
        """
        Safely extract individual Track objects from a list that may contain Track or TrackGroup objects.
        For TrackGroups, returns the best track from each group.
        """
        tracks = []
        for item in tracks_or_groups:
            # Check if it's a TrackGroup (has tracks attribute)
            if hasattr(item, 'tracks') and hasattr(item, 'get_best'):
                # It's a TrackGroup, get best track
                tracks.append(item.get_best())
            elif hasattr(item, 'weakhash'):
                # It's already a Track object
                tracks.append(item)
        return tracks

    def _get_cache(self, cache: OrderedDict, key: str) -> Optional[Any]:
        """Get item from cache with TTL check."""
        self._cache_requests += 1

        if key in cache:
            item, timestamp = cache[key]
            if time.time() - timestamp < self.CACHE_TTL:
                # Move to end (most recently used)
                cache.move_to_end(key)
                self._cache_hits += 1
                return item
            else:
                # Expired, remove it
                del cache[key]
        return None

    def _set_cache(self, cache: OrderedDict, key: str, value: Any):
        """Set item in cache with LRU eviction."""
        cache[key] = (value, time.time())
        cache.move_to_end(key)

        # Evict oldest if cache is full
        if len(cache) > self.MAX_CACHE_SIZE:
            cache.popitem(last=False)

    def _calculate_diversity_score(self, item_hash: str, recent_items: Set[str]) -> float:
        """Calculate diversity score to avoid repetitive recommendations."""
        if not recent_items:
            return 1.0

        # Get artist hashes for diversity calculation
        track = self.track_store.trackhashmap.get(item_hash)
        if not track:
            return 0.5

        item_artists = set(track.artisthashes)

        # Check how many recent recommendations share artists
        overlapping_artists = 0
        for recent_hash in recent_items:
            recent_track = self.track_store.trackhashmap.get(recent_hash)
            if recent_track and item_artists.intersection(recent_track.artisthashes):
                overlapping_artists += 1

        # Diversity score decreases with more overlaps
        diversity_penalty = min(overlapping_artists * 0.2, 0.8)
        return max(0.2, 1.0 - diversity_penalty)

    def _build_user_profile(self, userid: int) -> Dict[str, Any]:
        """Build user listening profile for collaborative filtering."""
        if userid in self._user_profile_cache:
            return self._user_profile_cache[userid]

        # Get user's recent listening history (last 90 days)
        ninety_days_ago = time.time() - (90 * 24 * 60 * 60)

        user_scrobbles = list(ScrobbleTable.get_all_in_period(
            ninety_days_ago, time.time(), userid
        ))

        if not user_scrobbles:
            profile = {
                'top_artists': [],
                'top_genres': [],
                'listening_patterns': {},
                'total_scrobbles': 0
            }
        else:
            # Analyze listening patterns
            artist_counts = Counter()
            genre_counts = Counter()
            track_counts = Counter()

            for scrobble in user_scrobbles:
                track = self.track_store.trackhashmap.get(scrobble.trackhash)
                if track:
                    track_counts[scrobble.trackhash] += 1

                    for artisthash in track.artisthashes:
                        artist_counts[artisthash] += 1

                    if track.genrehashes:
                        for genre_hash in track.genrehashes:
                            genre_counts[genre_hash] += 1

            profile = {
                'top_artists': [artist for artist, _ in artist_counts.most_common(10)],
                'top_genres': [genre for genre, _ in genre_counts.most_common(5)],
                'top_tracks': [track for track, _ in track_counts.most_common(20)],
                'total_scrobbles': len(user_scrobbles),
                'avg_daily_scrobbles': len(user_scrobbles) / 90
            }

        self._user_profile_cache[userid] = profile
        return profile

    def get_user_based_recommendations(self, userid: int, limit: int = 20) -> List[RecommendationResult]:
        """Get collaborative filtering recommendations based on similar users."""
        user_profile = self._build_user_profile(userid)

        if not user_profile['top_artists']:
            return []

        # Find similar users based on artist preferences
        similar_users = self._find_similar_users(user_profile['top_artists'])

        # Get tracks that similar users like but this user hasn't listened to much
        recommendations = []

        for similar_user_id in similar_users[:5]:  # Top 5 similar users
            similar_profile = self._build_user_profile(similar_user_id)

            for track_hash in similar_profile['top_tracks']:
                if track_hash not in user_profile['top_tracks']:
                    # Calculate recommendation score
                    track = self.track_store.trackhashmap.get(track_hash)
                    if track:
                        score = self._calculate_user_recommendation_score(
                            track, user_profile, similar_profile
                        )

                        if score > 0:
                            similarity_score = SimilarityScore(
                                item_hash=track_hash,
                                score=score,
                                reasons=["collaborative_filtering"],
                                confidence=0.7
                            )
                            recommendations.append(RecommendationResult(
                                item_hash=track_hash,
                                score=similarity_score,
                                reasons=["collaborative_filtering"],
                                confidence=0.7,
                                item_type="track"
                            ))

        # Sort and deduplicate
        seen_tracks = set()
        unique_recommendations = []

        for rec in sorted(recommendations, key=lambda x: x.score, reverse=True):
            if rec.item_hash not in seen_tracks:
                seen_tracks.add(rec.item_hash)
                unique_recommendations.append(rec)
                if len(unique_recommendations) >= limit:
                    break

        return unique_recommendations

    def _find_similar_users(self, user_artists: List[str], limit: int = 10) -> List[int]:
        """Find users with similar artist preferences."""
        # This is a simplified implementation
        # In production, you'd want to use more sophisticated similarity metrics
        similar_users = []

        # For now, return some random users as similar
        # TODO: Implement proper user similarity calculation
        from swingmusic.db.userdata import UserTable
        all_users = [user.id for user in UserTable.get_all()]

        # Remove current user (assuming we know it)
        # For now, just return a subset
        return all_users[:min(limit, len(all_users))]

    def _calculate_user_recommendation_score(self, track: Track, user_profile: Dict, similar_profile: Dict) -> float:
        """Calculate how well a track fits a user's taste based on collaborative data."""
        score = 0.0

        # Check if track's artists are in user's top artists
        for artisthash in track.artisthashes:
            if artisthash in user_profile['top_artists']:
                score += 2.0
            elif artisthash in similar_profile['top_artists']:
                score += 1.0

        # Check genre alignment
        if track.genrehashes:
            for genre_hash in track.genrehashes:
                if genre_hash in user_profile['top_genres']:
                    score += 1.5

        return score

    def get_similar_tracks(self, seed_tracks: List[Any], limit: int = 50, userid: Optional[int] = None) -> List[str]:
        """
        Get similar tracks based on seed tracks with enhanced algorithms.

        Args:
            seed_tracks: List of seed tracks/groups to base recommendations on (Track or TrackGroup objects)
            limit: Maximum number of similar tracks to return
            userid: Optional user ID for personalized recommendations

        Returns:
            List of weakhashes of similar tracks
        """
        if not seed_tracks:
            return []

        # Combine content-based and collaborative filtering
        all_recommendations = []

        # Content-based filtering (existing logic)
        content_scores = self._get_content_based_recommendations(seed_tracks, limit * 2)

        # Convert to RecommendationResult objects
        for track_hash, score_data in content_scores.items():
            similarity_score = SimilarityScore(
                item_hash=track_hash,
                score=score_data['score'],
                reasons=score_data['reasons'],
                confidence=score_data['confidence']
            )
            all_recommendations.append(RecommendationResult(
                item_hash=track_hash,
                score=similarity_score,
                reasons=score_data['reasons'],
                confidence=score_data['confidence'],
                item_type="track"
            ))

        # Add collaborative filtering if user provided
        if userid:
            collaborative_recs = self.get_user_based_recommendations(userid, limit // 2)
            all_recommendations.extend(collaborative_recs)

        # Apply diversity and temporal boosting
        recent_items = self._recent_recommendations.get(str(userid), set()) if userid else set()

        for rec in all_recommendations:
            # Calculate diversity score
            rec.score.diversity_score = self._calculate_diversity_score(rec.item_hash, recent_items)

            # Add temporal boost (recently popular tracks get slight boost)
            track = self.track_store.trackhashmap.get(rec.item_hash)
            if track:
                # Simple recency boost based on when track was added/modified
                rec.score.temporal_boost = min(track.playcount / 1000, 0.5)  # Cap at 0.5

        # Sort by final weighted score
        all_recommendations.sort(key=lambda x: x.score.final_score(), reverse=True)

        # Filter out seed tracks and apply diversity
        seed_weakhashes = self._extract_weakhashes(seed_tracks)
        result = []
        seen_artists = set()

        for rec in all_recommendations:
            track = self.track_store.trackhashmap.get(rec.item_hash)
            if not track or track.weakhash in seed_weakhashes:
                continue

            # Diversity check: avoid too many tracks from same artist
            track_artists = set(track.artisthashes)
            if track_artists.intersection(seen_artists) and len(result) >= limit // 2:
                continue  # Skip if we've seen this artist and have enough results

            result.append(track.weakhash)
            seen_artists.update(track_artists)

            # Track recent recommendations for diversity
            if userid:
                self._recent_recommendations[str(userid)].add(rec.item_hash)
                # Keep only recent 100 recommendations
                if len(self._recent_recommendations[str(userid)]) > 100:
                    self._recent_recommendations[str(userid)].pop()

            if len(result) >= limit:
                break

        return result

    def _get_content_based_recommendations(self, seed_tracks: List[Any], limit: int) -> Dict[str, Dict[str, Any]]:
        """Get content-based recommendations with enhanced scoring."""
        all_scores: Dict[str, Dict[str, Any]] = {}

        # Extract individual Track objects from TrackGroups if needed
        actual_tracks = self._extract_tracks(seed_tracks)

        for seed_track in actual_tracks:
            # Check cache first
            cache_key = seed_track.trackhash
            cached_result = self._get_cache(self._track_similarity_cache, cache_key)

            if cached_result is None:
                cached_result = self._get_similar_tracks_for_track(seed_track, limit=100)
                self._set_cache(self._track_similarity_cache, cache_key, cached_result)

            for score_obj in cached_result:
                track_hash = score_obj.item_hash

                if track_hash not in all_scores:
                    all_scores[track_hash] = {
                        'score': 0.0,
                        'count': 0,
                        'reasons': [],
                        'confidence': 0.0
                    }

                # Aggregate scores
                all_scores[track_hash]['score'] += score_obj.score
                all_scores[track_hash]['count'] += 1
                all_scores[track_hash]['reasons'].extend(score_obj.reasons)
                all_scores[track_hash]['confidence'] = max(
                    all_scores[track_hash]['confidence'],
                    score_obj.confidence or 0.8
                )

        # Boost scores for tracks similar to multiple seed tracks
        for track_hash, data in all_scores.items():
            if data['count'] > 1:
                data['score'] *= (1 + 0.1 * data['count'])
                data['confidence'] = min(data['confidence'] + 0.1 * data['count'], 1.0)

        # Sort and return top results
        sorted_tracks = sorted(all_scores.items(), key=lambda x: x[1]['score'], reverse=True)
        return dict(sorted_tracks[:limit])

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
        Get similar artists based on seed tracks with enhanced algorithms.

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

    def get_hybrid_recommendations(self, seed_tracks: List[Any], userid: Optional[int] = None,
                                 limit: int = 50) -> List[RecommendationResult]:
        """
        Get hybrid recommendations combining multiple strategies.

        Args:
            seed_tracks: Seed tracks/groups for content-based filtering (Track or TrackGroup objects)
            userid: User ID for collaborative filtering
            limit: Maximum recommendations to return

        Returns:
            List of RecommendationResult objects
        """
        recommendations = []

        # Strategy 1: Content-based filtering (40% of results)
        content_limit = int(limit * 0.4)
        if seed_tracks:
            content_recs = []
            content_scores = self._get_content_based_recommendations(seed_tracks, content_limit * 2)

            for track_hash, score_data in list(content_scores.items())[:content_limit]:
                content_recs.append(RecommendationResult(
                    item_hash=track_hash,
                    score=SimilarityScore(
                        item_hash=track_hash,
                        score=score_data['score'],
                        reasons=score_data['reasons'],
                        confidence=score_data['confidence']
                    ),
                    reasons=score_data['reasons'],
                    confidence=score_data['confidence'],
                    item_type="track"
                ))
            recommendations.extend(content_recs)

        # Strategy 2: Collaborative filtering (30% of results)
        collab_limit = int(limit * 0.3)
        if userid:
            collab_recs = self.get_user_based_recommendations(userid, collab_limit)
            recommendations.extend(collab_recs)

        # Strategy 3: Popularity-based discovery (20% of results)
        popular_limit = int(limit * 0.2)
        popular_recs = self._get_popular_discovery_tracks(userid, popular_limit)
        recommendations.extend(popular_recs)

        # Strategy 4: Serendipity/exploration (10% of results)
        serendipity_limit = limit - len(recommendations)
        if serendipity_limit > 0:
            serendipity_recs = self._get_serendipity_recommendations(userid, serendipity_limit)
            recommendations.extend(serendipity_recs)

        # Apply final ranking and diversity
        return self._finalize_recommendations(recommendations, userid, limit)

    def _get_popular_discovery_tracks(self, userid: Optional[int], limit: int) -> List[RecommendationResult]:
        """Get popular tracks the user hasn't listened to."""
        if not userid:
            return []

        user_profile = self._build_user_profile(userid)
        listened_tracks = set(user_profile['top_tracks'])

        # Get globally popular tracks
        all_tracks = self.track_store.get_flat_list()
        popular_tracks = sorted(all_tracks, key=lambda x: x.playcount, reverse=True)

        recommendations = []
        for track in popular_tracks[:limit * 3]:  # Get more candidates
            if track.trackhash not in listened_tracks:
                recommendations.append(RecommendationResult(
                    item_hash=track.trackhash,
                    score=SimilarityScore(
                        item_hash=track.trackhash,
                        score=min(track.playcount / 100, 2.0),  # Normalize popularity score
                        reasons=["global_popularity"],
                        confidence=0.6
                    ),
                    reasons=["global_popularity"],
                    confidence=0.6,
                    item_type="track"
                ))

        return recommendations[:limit]

    def _get_serendipity_recommendations(self, userid: Optional[int], limit: int) -> List[RecommendationResult]:
        """Get serendipitous recommendations for exploration."""
        all_tracks = self.track_store.get_flat_list()

        # Random selection with some bias toward less popular tracks
        candidates = [t for t in all_tracks if t.playcount < 10]  # Less played tracks

        if len(candidates) < limit:
            candidates = all_tracks

        selected = random.sample(candidates, min(limit, len(candidates)))

        recommendations = []
        for track in selected:
            recommendations.append(RecommendationResult(
                item_hash=track.trackhash,
                score=SimilarityScore(
                    item_hash=track.trackhash,
                    score=0.5,  # Base serendipity score
                    reasons=["serendipity"],
                    confidence=0.3
                ),
                reasons=["serendipity"],
                confidence=0.3,
                item_type="track"
            ))

        return recommendations

    def _finalize_recommendations(self, recommendations: List[RecommendationResult],
                                userid: Optional[int], limit: int) -> List[RecommendationResult]:
        """Apply final ranking, diversity, and filtering."""
        if not recommendations:
            return []

        # Apply diversity and temporal boosting
        recent_items = self._recent_recommendations.get(str(userid), set()) if userid else set()

        for rec in recommendations:
            rec.score.diversity_score = self._calculate_diversity_score(rec.item_hash, recent_items)
            track = self.track_store.trackhashmap.get(rec.item_hash)
            if track:
                rec.score.temporal_boost = min(track.playcount / 1000, 0.5)

        # Sort by final score
        recommendations.sort(key=lambda x: x.score.final_score(), reverse=True)

        # Apply diversity filtering
        final_recommendations = []
        seen_artists = set()

        for rec in recommendations:
            if len(final_recommendations) >= limit:
                break

            track = self.track_store.trackhashmap.get(rec.item_hash)
            if not track:
                continue

            track_artists = set(track.artisthashes)

            # Allow some artist repetition but not too much
            if len(final_recommendations) < limit // 2 or not track_artists.intersection(seen_artists):
                final_recommendations.append(rec)
                seen_artists.update(track_artists)

                # Track for future diversity
                if userid:
                    self._recent_recommendations[str(userid)].add(rec.item_hash)
                    if len(self._recent_recommendations[str(userid)]) > 100:
                        # Remove oldest
                        self._recent_recommendations[str(userid)].pop()

        return final_recommendations

    def _get_similar_tracks_for_track(self, track: Track, limit: int = 50) -> List[SimilarityScore]:
        """
        Get similar tracks for a single track based on enhanced content similarity.
        """
        cache_key = track.trackhash
        cached_result = self._get_cache(self._track_similarity_cache, cache_key)
        if cached_result is not None:
            return cached_result

        scores: List[SimilarityScore] = []

        # Use heap to efficiently track top N scores
        min_heap = []
        tie_breaker = 0  # For handling identical scores

        # Get all tracks for comparison
        all_tracks = self.track_store.get_flat_list()

        for other_track in all_tracks:
            if other_track.trackhash == track.trackhash:
                continue

            score = self._calculate_track_similarity(track, other_track)
            if score.score > 0:
                score.confidence = self._calculate_confidence(score)
                final_score = score.final_score()

                # Use heap to maintain top N scores (min-heap for largest N)
                heap_item = (final_score, tie_breaker, score)
                tie_breaker += 1

                if len(min_heap) < limit:
                    heapq.heappush(min_heap, heap_item)
                elif final_score > min_heap[0][0]:
                    heapq.heappop(min_heap)
                    heapq.heappush(min_heap, heap_item)

        # Extract results from heap, sorted by score descending
        sorted_scores = [score for _, _, score in sorted(min_heap, key=lambda x: (-x[0], x[1]))]
        self._set_cache(self._track_similarity_cache, cache_key, sorted_scores)

        return sorted_scores

    def _calculate_confidence(self, score: SimilarityScore) -> float:
        """Calculate confidence score based on similarity reasons."""
        confidence = 0.5  # Base confidence

        # Higher confidence for multiple strong reasons
        reason_count = len(score.reasons)
        if reason_count > 1:
            confidence += min(reason_count * 0.1, 0.3)

        # Higher confidence for strong similarity scores
        if score.score > 2.0:
            confidence += 0.2

        return min(confidence, 1.0)

    def _calculate_track_similarity(self, track1: Track, track2: Track) -> SimilarityScore:
        """
        Calculate enhanced similarity score between two tracks.
        """
        score = 0.0
        reasons = []

        # Artist similarity (most important)
        artist_similarity = self._calculate_artist_overlap(track1, track2)
        if artist_similarity > 0:
            score += artist_similarity * 3.0
            reasons.append(f"artist_overlap:{artist_similarity:.2f}")

        # Genre similarity with weighted scoring
        genre_similarity = self._calculate_genre_overlap(track1, track2)
        if genre_similarity > 0:
            score += genre_similarity * 2.0
            reasons.append(f"genre_overlap:{genre_similarity:.2f}")

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

        # BPM/Tempo similarity (if available)
        if hasattr(track1, 'bpm') and hasattr(track2, 'bpm') and track1.bpm and track2.bpm:
            bpm_diff = abs(track1.bpm - track2.bpm)
            if bpm_diff < 20:  # Within 20 BPM
                bpm_similarity = 1.0 - (bpm_diff / 20.0)
                score += bpm_similarity * 0.3
                reasons.append(f"bpm_similarity:{bpm_similarity:.2f}")

        # Year/Release date similarity (music from similar eras)
        if hasattr(track1, 'year') and hasattr(track2, 'year') and track1.year and track2.year:
            year_diff = abs(track1.year - track2.year)
            if year_diff <= 5:  # Within 5 years
                year_similarity = 1.0 - (year_diff / 5.0)
                score += year_similarity * 0.2
                reasons.append(f"era_similarity:{year_similarity:.2f}")

        # Popularity correlation (tracks with similar play counts might be similar quality)
        if hasattr(track1, 'playcount') and hasattr(track2, 'playcount'):
            playcount_ratio = min(track1.playcount, track2.playcount) / max(track1.playcount, track2.playcount)
            if playcount_ratio > 0.1:  # Not too different in popularity
                score += playcount_ratio * 0.1
                reasons.append(f"popularity_correlation:{playcount_ratio:.2f}")

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
        Get similar artists for an artist using stored LastFM data with caching.
        """
        cached_result = self._get_cache(self._artist_similarity_cache, artisthash)
        if cached_result is not None:
            return cached_result

        # Try to get from database first
        similar_artist_data = SimilarArtistTable.get_by_hash(artisthash)
        if similar_artist_data:
            # Convert to list of hashes and cache
            similar_hashes = list(similar_artist_data.similar_artists.keys())
            result = similar_hashes[:limit]
            self._set_cache(self._artist_similarity_cache, artisthash, result)
            return result

        # Fallback: find artists with similar genres
        artist = self.artist_store.artistmap.get(artisthash)
        if not artist:
            empty_result = []
            self._set_cache(self._artist_similarity_cache, artisthash, empty_result)
            return empty_result

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

        # Sort by overlap strength (simplified)
        result = similar_hashes[:limit]
        self._set_cache(self._artist_similarity_cache, artisthash, result)
        return result


    def clear_cache(self):
        """Clear all caches (useful for testing or memory management)."""
        self._track_similarity_cache.clear()
        self._artist_similarity_cache.clear()
        self._user_profile_cache.clear()
        self._recent_recommendations.clear()

    def get_cache_stats(self) -> Dict[str, int]:
        """Get statistics about cache usage."""
        return {
            'track_similarity_cache_size': len(self._track_similarity_cache),
            'artist_similarity_cache_size': len(self._artist_similarity_cache),
            'user_profile_cache_size': len(self._user_profile_cache),
            'recent_recommendations_tracked': sum(len(items) for items in self._recent_recommendations.values())
        }

    def precompute_popular_tracks(self, limit: int = 1000):
        """Precompute most popular tracks for faster access."""
        all_tracks = self.track_store.get_flat_list()
        self._popular_tracks = sorted(all_tracks, key=lambda x: x.playcount, reverse=True)[:limit]

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for monitoring."""
        return {
            'cache_stats': self.get_cache_stats(),
            'total_tracks': len(self.track_store.get_flat_list()) if hasattr(self.track_store, 'get_flat_list') else 0,
            'total_artists': len(self.artist_store.artistmap) if hasattr(self.artist_store, 'artistmap') else 0,
            'cache_hit_rate': getattr(self, '_cache_hits', 0) / max(getattr(self, '_cache_requests', 1), 1)
        }


# Global instance for use throughout the application
offline_engine = OfflineRecommendationEngine()
