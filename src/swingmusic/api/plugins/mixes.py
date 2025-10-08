from typing import Literal
from flask_openapi3 import Tag
from flask_openapi3 import APIBlueprint
from pydantic import BaseModel, Field

from swingmusic.db.userdata import MixTable
from swingmusic.plugins.mixes import MixesPlugin
from swingmusic.store.homepage import HomepageStore
from swingmusic.store.tracks import TrackStore
from swingmusic.lib.recipes.artistmixes import ArtistMixes
from swingmusic.lib.recipes.because import BecauseYouListened


bp_tag = Tag(name="Mixes Plugin", description="Mixes plugin hehe")
api = APIBlueprint(
    "mixesplugin", __name__, url_prefix="/plugins/mixes", abp_tags=[bp_tag]
)


class GetMixesBody(BaseModel):
    mixtype: Literal["artists", "tracks"] = Field(description="The type of mix")


@api.get("/<mixtype>")
def get_artist_mixes(path: GetMixesBody):
    srcmixes = MixTable.get_all(with_userid=True)
    mixes = []

    if path.mixtype == "artists":
        mixes = [mix.to_dict(convert_timestamp=True) for mix in srcmixes]
    elif path.mixtype == "tracks":
        plugin = MixesPlugin()

        for mix in srcmixes:
            custom_mix = plugin.get_track_mix(mix)
            if custom_mix:
                mixes.append(custom_mix.to_dict(convert_timestamp=True))

    seen_mixids = set()

    # filter duplicates by trackshash
    final_mixes = []
    for mix in mixes:
        # INFO: Ignore duplicates for artist mixes
        if mix["id"] in seen_mixids and path.mixtype == "tracks":
            continue

        final_mixes.append(mix)
        seen_mixids.add(mix["id"])

    return final_mixes


class MixQuery(BaseModel):
    mixid: str = Field(description="The mix id")
    sourcehash: str = Field(description="The sourcehash of the mix")


@api.get("/")
def get_mix(query: MixQuery):
    mixtype = ""

    match query.mixid[0]:
        case "a":
            mixtype = "artist_mixes"
        case "t":
            mixtype = "custom_mixes"
        case _:
            return {"msg": "Invalid mix ID"}, 400

    # INFO: Check if the mix is already in the homepage store
    mix = HomepageStore.get_mix(mixtype, query.mixid)
    if mix and mix["sourcehash"] == query.sourcehash:
        return mix, 200

    # INF0: Get the mix from the db
    mix = MixTable.get_by_sourcehash(query.sourcehash)

    if not mix:
        return {"msg": "Mix not found"}, 404

    if mixtype == "custom_mixes":
        mix = MixesPlugin.get_track_mix(mix)

        if not mix:
            return {"msg": "Mix not found"}, 404

    return mix.to_full_dict(), 200


class SaveMixRequest(BaseModel):
    mixid: str = Field(description="The id of the mix")
    type: str = Field(description="The type of mix")
    sourcehash: str = Field(description="The sourcehash of the mix")


@api.post("/save")
def save_mix(body: SaveMixRequest):
    mix_type = body.type
    mix_sourcehash = body.sourcehash

    if mix_type == "artist":
        state = MixTable.save_artist_mix(mix_sourcehash)
    elif mix_type == "track":
        state = MixTable.save_track_mix(mix_sourcehash)

    mix = HomepageStore.find_mix(body.mixid)

    if mix:
        mix.saved = state
    return {"msg": "Mixes saved"}, 200


@api.post("/generate")
def generate_mixes():
    """
    Manually trigger mix generation (artist mixes + recommendations).
    This runs the same process as the 12-hour cron job.
    """
    try:
        ArtistMixes()
        BecauseYouListened()
        return {"msg": "Mixes generated successfully"}, 200
    except Exception as e:
        return {"msg": f"Failed to generate mixes: {str(e)}"}, 500
