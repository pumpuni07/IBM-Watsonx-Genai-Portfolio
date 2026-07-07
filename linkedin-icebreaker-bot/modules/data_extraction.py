"""LinkedIn profile data extraction: ProxyCurl API or local mock data.

Deviation from the lab (disclosed): the lab loads mock data from a remote
URL baked into its starter tarball; this version ships the mock profile
in the repo and reads it locally, since ProxyCurl itself was discontinued
in February 2025 and the mock path is the working one.
"""

import json
import logging
import time
from typing import Any, Dict, Optional

import requests

import config

logger = logging.getLogger(__name__)


def extract_linkedin_profile(
    linkedin_profile_url: str,
    api_key: Optional[str] = None,
    mock: bool = False,
) -> Dict[str, Any]:
    """Extract LinkedIn profile data using the API or a local mock JSON file."""
    start_time = time.time()

    try:
        if mock:
            logger.info("Using mock data from local JSON file...")
            with open(config.MOCK_DATA_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            if not api_key:
                raise ValueError("API key is required when mock is set to False.")

            logger.info("Starting to extract the LinkedIn profile...")
            api_endpoint = "https://nubela.co/proxycurl/api/v2/linkedin"
            headers = {"Authorization": f"Bearer {api_key}"}
            params = {
                "url": linkedin_profile_url,
                "fallback_to_cache": "on-error",
                "use_cache": "if-present",
                "skills": "include",
            }
            logger.info(f"Sending API request at {time.time() - start_time:.2f}s...")
            response = requests.get(api_endpoint, headers=headers, params=params, timeout=10)
            if response.status_code != 200:
                logger.error(f"Failed to retrieve data. Status code: {response.status_code}")
                return {}
            data = response.json()

        # Clean the data: remove empty values and unwanted fields
        data = {
            k: v
            for k, v in data.items()
            if v not in ([], "", None) and k not in ["people_also_viewed", "certifications"]
        }
        if data.get("groups"):
            for group_dict in data.get("groups"):
                group_dict.pop("profile_pic_url", None)

        logger.info(f"Profile data ready at {time.time() - start_time:.2f}s")
        return data

    except Exception as e:
        logger.error(f"Error in extract_linkedin_profile: {e}")
        return {}
