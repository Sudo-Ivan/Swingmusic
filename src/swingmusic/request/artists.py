"""
Requests related to artists
"""

import urllib.parse
import logging

import requests
from requests import ConnectionError, HTTPError, ReadTimeout

from swingmusic.models.lastfm import SimilarArtistEntry
from swingmusic.utils.hashing import create_hash

log = logging.getLogger(__name__)


def fetch_similar_artists(name: str):
    """
    Fetches similar artists from Last.fm
    """
    url = f"https://kerve.last.fm/kerve/similarartists?artist={urllib.parse.quote_plus(name, safe='')}&autocorrect=1&tracks=1&image_size=large&limit=250&format=json"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except ConnectionError as e:
        log.warning(f"Connection error fetching similar artists for '{name}': {e}")
        return None
    except ReadTimeout as e:
        log.warning(f"Timeout fetching similar artists for '{name}': {e}")
        return None
    except HTTPError as e:
        log.warning(f"HTTP error fetching similar artists for '{name}': {e}")
        return None
    except Exception as e:
        log.error(f"Unexpected error fetching similar artists for '{name}': {e}")
        return None

    try:
        data = response.json()
    except Exception as e:
        log.error(f"Failed to parse JSON response for artist '{name}': {e}")
        return []

    try:
        artists = data["results"]["artist"]
    except KeyError:
        log.debug(f"No similar artists found for '{name}' (missing results.artist key)")
        return []

    try:
        return [
            SimilarArtistEntry(
               **{
                    "artisthash": create_hash(artist["name"]),
                    "name": artist["name"],
                    "weight": artist["weight"],
                    "listeners": int(artist["listeners"]),
                    "scrobbles": int(artist["scrobbles"]),
                }
            )
            for artist in artists
        ]
    except (KeyError, ValueError, TypeError) as e:
        log.error(f"Failed to parse artist data for '{name}': {e}")
        return []
