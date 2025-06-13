"""Tools to generate from OpenAI prompts.
Adopted from https://github.com/zeno-ml/zeno-build/"""

import asyncio
import logging
import os
import random
import time
import requests
from typing import Any

import aiolimiter
import openai
from openai import OpenAI
from tqdm.asyncio import tqdm_asyncio

from azure.identity import AzureCliCredential, ChainedTokenCredential, DefaultAzureCredential, ManagedIdentityCredential, get_bearer_token_provider
from msal import PublicClientApplication
from openai import OpenAI, AzureOpenAI
from PIL import Image
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)
AZURE_ENDPOINT = os.environ.get("AZURE_ENDPOINT", None)
SERVICE_LOGIN = os.environ.get("P24_LOGIN", "").lower().strip()
AZURE_OPENAI_ENDPOINT = os.environ.get("OPENAI_ENDPOINT")
MODEL = os.environ["OPENAI_VISION_MODEL"]
ENDPOINT_TYPE = os.environ["OPENAI_ENDPOINT_TYPE"]
SCOPE = os.environ.get("P24_SCOPE", "https://cognitiveservices.azure.com/.default")
API_VERSION = "2025-01-01-preview"
LOGGER = logging.getLogger(__name__)

headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {OPENAI_API_KEY}"
}
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

def ping_server(url, token):
    url = f"{url}/ping/"
    headers = {"Authorization": "Bearer " + token}

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.text
    else:
        print(response.text)
        response.raise_for_status()

def msal_login():
    msal_token = os.environ.get("MSAL_TOKEN", "")
    if msal_token:
        # I am not sure why but the token is sometimes prefixed with "env:"
        if msal_token.startswith("env:"):
            msal_token = msal_token[5:]
        LOGGER.warning("Using cached token")
        return msal_token

    client_id = os.environ["P24_CLIENT_ID"]
    authority = os.environ["P24_AUTHORITY"]
    scopes = [os.environ["P24_SCOPE"]]
    url = os.environ["OPENAI_ENDPOINT"]

    app = PublicClientApplication(
        client_id,
        authority=authority,
    )

    # initialize result variable to hole the token response
    result = None

    # We now check the cache to see
    # whether we already have some accounts that the end user already used to sign in before.
    accounts = app.get_accounts()
    if accounts:
        # If so, you could then somehow display these accounts and let end user choose
        print("Pick the number of the account you want to use to proceed:")
        for i, a in enumerate(accounts):
            print(f"ID: {i} {a['username']}")
        # Assuming the end user chose this one
        chosen = accounts[int(input("Chose the account number: "))]
        # Now let's try to find a token in cache for this account
        result = app.acquire_token_silent(scopes, account=chosen)

    if not result:
        # So no suitable token exists in cache. Let's get a new one from Azure AD.
        result = app.acquire_token_interactive(scopes=scopes)
    if "access_token" in result:
        LOGGER.info("successfully logged in")  # Yay!
        access_token = result["access_token"]
        ping_server(url=url, token=access_token)
        return access_token
    else:
        LOGGER.error(result.get("error"))
        LOGGER.error(result.get("error_description"))
        LOGGER.error(result.get("correlation_id"))  # You may need this when reporting a bug
        raise Exception("Login failed")

def get_managed_identity_token():
    token = ManagedIdentityCredential(client_id=os.environ.get("AZURE_CLIENT_ID", "")).get_token(
        "https://cognitiveservices.azure.com/.default"
    )
    return token.token

def get_azure_ad_provider():
    scope = os.environ.get("P24_SCOPE", "https://cognitiveservices.azure.com/.default")
    token_provider = get_bearer_token_provider(
        ChainedTokenCredential(
            AzureCliCredential(),
            DefaultAzureCredential(
                exclude_cli_credential=True,
                # Exclude other credentials we are not interested in.
                exclude_environment_credential=True,
                exclude_shared_token_cache_credential=True,
                exclude_developer_cli_credential=True,
                exclude_powershell_credential=True,
                exclude_interactive_browser_credential=True,
                exclude_visual_studio_code_credentials=True,
            ),
        ),
        scope,
    )
    return token_provider

def retry_with_exponential_backoff(  # type: ignore
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 3,
    errors: tuple[Any] = (openai.RateLimitError,),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):  # type: ignore
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)
            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())
                print(f"Retrying in {delay} seconds.")
                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


async def _throttled_openai_completion_acreate(
    engine: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    limiter: aiolimiter.AsyncLimiter,
) -> dict[str, Any]:
    async with limiter:
        for _ in range(3):
            try:
                return await openai.Completion.acreate(  # type: ignore
                    engine=engine,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
            except openai.RateLimitError:
                logging.warning(
                    "OpenAI API rate limit exceeded. Sleeping for 10 seconds."
                )
                await asyncio.sleep(10)
            except openai.APIError as e:
                logging.warning(f"OpenAI API error: {e}")
                break
        return {"choices": [{"message": {"content": ""}}]}


async def agenerate_from_openai_completion(
    prompts: list[str],
    engine: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    requests_per_minute: int = 300,
) -> list[str]:
    """Generate from OpenAI Completion API.

    Args:
        prompts: list of prompts
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use.
        context_length: Length of context to use.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses.
    """
    #if "OPENAI_API_KEY" not in os.environ:
    #    raise ValueError(
    #        "OPENAI_API_KEY environment variable must be set when using OpenAI API. (v1)"
    #    )
    #openai.api_key = os.environ["OPENAI_API_KEY"]
    openai.organization = os.environ.get("OPENAI_ORGANIZATION", "")
    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_openai_completion_acreate(
            engine=engine,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
        )
        for prompt in prompts
    ]
    responses = await tqdm_asyncio.gather(*async_responses)
    return [x["choices"][0]["text"] for x in responses]


@retry_with_exponential_backoff
def generate_from_openai_completion(
    prompt: str,
    engine: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    stop_token: str | None = None,
) -> str:
    #if "OPENAI_API_KEY" not in os.environ:
    #    raise ValueError(
    #        "OPENAI_API_KEY environment variable must be set when using OpenAI API. (v2)"
    #    )
    #openai.api_key = os.environ["OPENAI_API_KEY"]
    openai.organization = os.environ.get("OPENAI_ORGANIZATION", "")
    response = openai_client.completions.create(  # type: ignore
        prompt=prompt,
        engine=engine,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stop=[stop_token],
    )
    answer: str = response["choices"][0]["text"]
    return answer


async def _throttled_openai_chat_completion_acreate(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    limiter: aiolimiter.AsyncLimiter,
) -> dict[str, Any]:
    async with limiter:
        for _ in range(3):
            try:
                return await openai.ChatCompletion.acreate(  # type: ignore
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
            except openai.RateLimitError:
                logging.warning(
                    "OpenAI API rate limit exceeded. Sleeping for 10 seconds."
                )
                await asyncio.sleep(10)
            except asyncio.exceptions.TimeoutError:
                logging.warning("OpenAI API timeout. Sleeping for 10 seconds.")
                await asyncio.sleep(10)
            except openai.APIError as e:
                logging.warning(f"OpenAI API error: {e}")
                break
        return {"choices": [{"message": {"content": ""}}]}


async def agenerate_from_openai_chat_completion(
    messages_list: list[list[dict[str, str]]],
    engine: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    requests_per_minute: int = 300,
) -> list[str]:
    """Generate from OpenAI Chat Completion API.

    Args:
        messages_list: list of message list
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use.
        context_length: Length of context to use.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses.
    """
    #if "OPENAI_API_KEY" not in os.environ:
    #    raise ValueError(
    #        "OPENAI_API_KEY environment variable must be set when using OpenAI API. (v3)"
    #    )
    #openai.api_key = os.environ["OPENAI_API_KEY"]
    openai.organization = os.environ.get("OPENAI_ORGANIZATION", "")
    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_openai_chat_completion_acreate(
            model=engine,
            messages=message,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
        )
        for message in messages_list
    ]
    responses = await tqdm_asyncio.gather(*async_responses)
    return [x["choices"][0]["message"]["content"] for x in responses]


@retry_with_exponential_backoff
def generate_from_openai_chat_completion(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    stop_token: str | None = None,
) -> str:
    #if "OPENAI_API_KEY" not in os.environ:
    #    raise ValueError(
    #        "OPENAI_API_KEY environment variable must be set when using OpenAI API. (v4)"
    #    )
    #openai.api_key = os.environ["OPENAI_API_KEY"]
    openai.organization = os.environ.get("OPENAI_ORGANIZATION", "")

    response = openai_client.chat.completions.create(  # type: ignore
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stop=[stop_token] if stop_token else None,
    )
    answer: str = response.choices[0].message.content
    return answer


@retry_with_exponential_backoff
# debug only
def fake_generate_from_openai_chat_completion(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    stop_token: str | None = None,
) -> str:
    #if "OPENAI_API_KEY" not in os.environ:
    #    raise ValueError(
    #        "OPENAI_API_KEY environment variable must be set when using OpenAI API. (v5)"
    #    )
    #openai.api_key = os.environ["OPENAI_API_KEY"]
    openai.organization = os.environ.get("OPENAI_ORGANIZATION", "")
    answer = "Let's think step-by-step. This page shows a list of links and buttons. There is a search box with the label 'Search query'. I will click on the search box to type the query. So the action I will perform is \"click [60]\"."
    return answer

def get_openai_client() -> OpenAI | AzureOpenAI:
    if SERVICE_LOGIN == "service":
        token = msal_login()
        return create_openai_client(azure_ad_token=token)
    elif SERVICE_LOGIN == "managed_identity":
        token = get_managed_identity_token()
        return create_openai_client(azure_ad_token=token)
    elif SERVICE_LOGIN == "bearer_token":
        token = os.environ["AZURE_AD_TOKEN"]
        return create_openai_client(azure_ad_token=token)
    elif SERVICE_LOGIN == "azure_ad_token_provider":
        token_provider = get_azure_ad_provider()
        return create_openai_client(azure_ad_token_provider=token_provider)
    else:
        api_key = os.environ["OPENAI_API_KEY"]
        return create_openai_client(api_key=api_key)

def create_openai_client(**kwargs) -> OpenAI | AzureOpenAI:
    if not ("api_key" in kwargs or "azure_ad_token" in kwargs or "azure_ad_token_provider" in kwargs):
        raise Exception("api_key or azure_ad_token or azure_ad_token_provider must be provided")

    if ENDPOINT_TYPE.lower() == "azure":
        return AzureOpenAI(azure_endpoint=AZURE_OPENAI_ENDPOINT, api_version=API_VERSION, **kwargs)
    else:
        return OpenAI(**kwargs)

openai_client = get_openai_client()
