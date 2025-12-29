"""
External API Integration for CognitiveLattice
Sends relevant chunks to external APIs (OpenAI, Claude, etc.) for enhanced analysis
"""

import os
import json
import requests
import httpx
import asyncio
import base64
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import time
from .exceptions import ApiConnectionError, ModelNotAvailableError, ConfigurationError
from .config import config
from .model_config import model_config
# Lazy loaded imports
# from google import genai
# from xai_sdk import Client
# import anthropic
# from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# Try to load environment variables, but don't fail if dotenv isn't available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logger.warning("python-dotenv not available, reading .env manually")
    # Manually read .env file if dotenv is not available
    env_path = ".env"
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value

class ExternalAPIClient:
    def __init__(self, api_provider="openai", api_key: Optional[str] = None):
        """
        Initialize ExternalAPIClient with API provider configuration.

        Args:
            api_provider: API provider to use ("openai", "gemini", "grok", "anthropic")
            api_key: Optional API key. If not provided, will look in environment.
        """
        self.api_provider = api_provider
        self.api_key = self._load_api_key(api_key)
        self.base_url = self._get_base_url()
        # Fetch available models - failure here should propagate
        # so LangGraph/callers know immediately if API is misconfigured
        self.available_models = self._fetch_available_models()
        
    def _load_api_key(self, provided_key: Optional[str] = None) -> str:
        """Load API key from argument or environment"""
        if self.api_provider == "openai":
            key = provided_key or os.getenv("OPENAI_API_KEY")
            if not key:
                raise ConfigurationError("OPENAI_API_KEY not found in arguments or environment")
            # Initialize async OpenAI client
            from openai import AsyncOpenAI
            self.async_openai_client = AsyncOpenAI(api_key=key)
            return key
        elif self.api_provider == "gemini":
            key = provided_key or os.getenv("GEMINI_API_KEY")
            if not key:
                raise ConfigurationError("GEMINI_API_KEY not found in arguments or environment")
            from google import genai
            self.genai_client = genai.Client(api_key=key)
            return key
        elif self.api_provider == "grok":
            key = provided_key or os.getenv("XAI_API_KEY")
            if not key:
                raise ConfigurationError("XAI_API_KEY not found in arguments or environment")
            return key
        elif self.api_provider == "anthropic":
            key = provided_key or os.getenv("ANTHROPIC_API_KEY")
            if not key:
                raise ConfigurationError("ANTHROPIC_API_KEY not found in arguments or environment")
            import anthropic
            self.anthropic_client = anthropic.Anthropic(api_key=key)
            # Initialize async Anthropic client
            self.async_anthropic_client = anthropic.AsyncAnthropic(api_key=key)
            return key
        else:
            raise ConfigurationError(f"Unsupported API provider: {self.api_provider}")
    
    def _get_base_url(self) -> str:
        """Get base URL for API provider"""
        if self.api_provider == "openai":
            return "https://api.openai.com/v1"
        elif self.api_provider == "gemini":
            return ""  # Gemini uses SDK, not REST base URL
        elif self.api_provider == "grok":
            return ""  # Grok uses SDK, not REST base URL
        elif self.api_provider == "anthropic":
            return ""  # Anthropic uses SDK, not REST base URL
        else:
            raise ConfigurationError(f"Unsupported API provider: {self.api_provider}")

    def _fetch_available_models(self) -> List[str]:
        """Fetch available model ids for this API key."""
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            resp = requests.get(f"{self.base_url}/models", headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return [m.get("id") for m in data.get("data", []) if m.get("id")]
        except Exception as e:
            logger.error(f"Could not fetch available models: {e}", exc_info=True)
            # Still return empty list but log as warning, 
            # as this shouldn't necessarily block everything but isn't "silent" anymore
            return []
    
    def query_external_api(self, query: str, max_tokens: int = None, model: str = None, **options) -> str:
        """
        Send a direct query to external API for simple questions and chat
        
        Args:
            query (str): The user's question or chat message
            max_tokens (int): Maximum tokens for response
            model (str): Model to use
            **options: Additional options (headers, timeout, provider_params, etc.)
                - headers (dict): Custom HTTP headers
                - timeout (int): Request timeout in seconds (default: 60)
                - temperature (float): Override default temperature
                - Additional provider-specific params passed through
        """
        max_tokens = max_tokens or model_config.MAX_RESPONSE_TOKENS
        model = model or model_config.get_main_model()
        
        # Ensure standard options have defaults if not provided
        options.setdefault('timeout', 60)
        options.setdefault('temperature', 0.1)
        
        try:
            logger.debug(f"Sending direct query to external API (model: {model}, options: {list(options.keys())})...")
            
            # Get current date for context
            from datetime import datetime
            current_date = datetime.now().strftime("%B %d, %Y")
            current_month = datetime.now().strftime("%B")
            
            messages = [
                {"role": "system", "content": f"You are a helpful AI assistant. Today's date is {current_date}. When answering questions about 'this time of year' or current conditions, use {current_month} {datetime.now().year} as the reference point. Provide clear, informative responses to user questions."},
                {"role": "user", "content": query}
            ]
            
            # Route to appropriate API based on provider
            if self.api_provider == "gemini":
                response_json = self._call_gemini_api(model, messages, max_tokens=max_tokens, **options)
            elif self.api_provider == "grok":
                response_json = self._call_grok_api(model, messages, max_tokens=max_tokens, **options)
            elif self.api_provider == "anthropic":
                response_json = self._call_anthropic_api(model, messages, max_tokens=max_tokens, **options)
            else:  # Default to OpenAI
                response_json = self._call_openai_api(model, messages, max_tokens=max_tokens, **options)

            # Extract content from normalized response shape
            try:
                content = response_json["choices"][0]["message"]["content"]
                # Ensure content is a string
                if not isinstance(content, str):
                    content = str(content)
            except Exception:
                # Best-effort fallback: dump the raw JSON if structure differs
                content = json.dumps(response_json)
                # Ensure content is always a string
                if not isinstance(content, str):
                    content = str(content)

            return content
            
        except Exception as e:
            logger.error(f"Direct query failed: {e}", exc_info=True)
            raise ApiConnectionError(f"Failed to connect to external API: {str(e)}") from e
    
    async def query_external_api_async(self, query: str, max_tokens: int = None, model: str = None, **options) -> str:
        """
        Async version: Send a direct query to external API for simple questions and chat
        
        Args:
            query (str): The user's question or chat message
            max_tokens (int): Maximum tokens for response
            model (str): Model to use
            **options: Additional options (headers, timeout, provider_params, etc.)
                - headers (dict): Custom HTTP headers
                - timeout (int): Request timeout in seconds (default: 60)
                - temperature (float): Override default temperature
                - Additional provider-specific params passed through
        """
        max_tokens = max_tokens or model_config.MAX_RESPONSE_TOKENS
        model = model or model_config.get_main_model()
        
        # Ensure standard options have defaults if not provided
        options.setdefault('timeout', 60)
        options.setdefault('temperature', 0.1)
        
        try:
            logger.debug(f"Sending async query to external API (model: {model}, options: {list(options.keys())})...")
            
            # Get current date for context
            from datetime import datetime
            current_date = datetime.now().strftime("%B %d, %Y")
            current_month = datetime.now().strftime("%B")
            
            messages = [
                {"role": "system", "content": f"You are a helpful AI assistant. Today's date is {current_date}. When answering questions about 'this time of year' or current conditions, use {current_month} {datetime.now().year} as the reference point. Provide clear, informative responses to user questions."},
                {"role": "user", "content": query}
            ]
            
            # Route to appropriate API based on provider
            if self.api_provider == "gemini":
                response_json = await self._call_gemini_api_async(model, messages, max_tokens=max_tokens, **options)
            elif self.api_provider == "grok":
                response_json = await self._call_grok_api_async(model, messages, max_tokens=max_tokens, **options)
            elif self.api_provider == "anthropic":
                response_json = await self._call_anthropic_api_async(model, messages, max_tokens=max_tokens, **options)
            else:  # Default to OpenAI
                response_json = await self._call_openai_api_async(model, messages, max_tokens=max_tokens, **options)

            # Extract content from normalized response shape
            try:
                content = response_json["choices"][0]["message"]["content"]
                # Ensure content is a string
                if not isinstance(content, str):
                    content = str(content)
            except Exception:
                # Best-effort fallback: dump the raw JSON if structure differs
                content = json.dumps(response_json)
                # Ensure content is always a string
                if not isinstance(content, str):
                    content = str(content)

            return content
            
        except Exception as e:
            logger.error(f"Async query failed: {e}", exc_info=True)
            raise ApiConnectionError(f"Failed to connect to external API: {str(e)}") from e
    
    def _call_openai_api(self, model: str, messages: List[Dict[str, Any]], max_tokens: int, **options) -> Dict[str, Any]:
        """Make API call to OpenAI with specified model and parameters"""
        temperature = options.get('temperature', 0.1)
        timeout = options.get('timeout', 60)
        custom_headers = options.get('headers', {})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        headers.update(custom_headers)

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        # Add provider-specific options to payload
        # Skip internal HMLR options that shouldn't go to OpenAI
        internal_options = {'temperature', 'timeout', 'headers'}
        for k, v in options.items():
            if k not in internal_options and k not in payload:
                payload[k] = v

        try:
            
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout
            )
            
            if resp.status_code == 400:
                try:
                    body = resp.json()
                    msg = body.get('error', {}).get('message', '')
                    if 'Unsupported parameter' in msg and 'max_tokens' in msg:
                        logger.debug("Retrying with 'max_completion_tokens'...")
                        import copy
                        adjusted = copy.deepcopy(payload)
                        adjusted['max_completion_tokens'] = adjusted.pop('max_tokens')
                        resp = requests.post(
                            f"{self.base_url}/chat/completions",
                            headers=headers,
                            json=adjusted,
                            timeout=timeout
                        )
                except Exception:
                    pass

            resp.raise_for_status()
            return resp.json()

        except Exception as e:
            logger.error(f"OpenAI API request failed: {e}", exc_info=True)
            raise ApiConnectionError(f"OpenAI API failure: {str(e)}")
    
    def _call_gemini_api(self, model: str, messages: List[Dict[str, Any]], max_tokens: int, **options) -> Dict[str, Any]:
        """Make API call to Google Gemini with specified model and parameters"""
        temperature = options.get('temperature', 0.1)
        

        try:
            prompt_parts = []
            for msg in messages:
                role = msg.get('role', '')
                content = msg.get('content', '')
                if role == 'system':
                    prompt_parts.append(f"System: {content}")
                elif role == 'user':
                    prompt_parts.append(f"User: {content}")
                elif role == 'assistant':
                    prompt_parts.append(f"Assistant: {content}")
            
            full_prompt = "\n\n".join(prompt_parts)
            
            # Prepare generation config
            gen_config = {
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            }
            
            # Add other options to gen_config (excluding internal ones)
            internal_options = {'temperature', 'timeout', 'headers'}
            for k, v in options.items():
                if k not in internal_options:
                    gen_config[k] = v

            # Prepare generation config
            from google import genai
            response = self.genai_client.models.generate_content(
                model=model,
                contents=full_prompt,
                config=genai.types.GenerateContentConfig(**gen_config)
            )
            
            logger.debug(f"Gemini response candidates: {len(response.candidates)}")
            if response.candidates:
                candidate = response.candidates[0]
                logger.debug(f"Finish reason: {candidate.finish_reason}")
                if hasattr(candidate, 'safety_ratings'):
                    logger.debug(f"Safety ratings: {candidate.safety_ratings}")
            
            # Extract text from response
            response_text = response.text
            logger.debug(f"Response length: {len(response_text)} characters")
            
            # Create normalized response structure matching OpenAI format
            normalized = {
                'choices': [
                    {'message': {'content': response_text}}
                ],
                'model': model,
                'usage': {
                    'prompt_tokens': response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0,
                    'completion_tokens': response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0,
                    'total_tokens': response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else 0,
                }
            }
            
            return normalized
            
        except Exception as e:
            logger.error(f"Gemini request failed: {e}", exc_info=True)
            raise
    
    def _call_grok_api(self, model: str, messages: List[Dict[str, Any]], max_tokens: int, **options) -> Dict[str, Any]:
        """Make API call to Grok (xAI) with specified model and parameters"""
        
        try:
            # Initialize the Grok client
            from xai_sdk import Client
            client = Client(api_key=self.api_key, timeout=3600)
            
            # Create a chat session
            chat = client.chat.create(model=model)
            
            # Add messages to the chat
            for msg in messages:
                role = msg.get('role', '')
                content = msg.get('content', '')
                
                if role == 'system':
                    from xai_sdk.chat import system
                    chat.append(system(content))
                elif role == 'user':
                    from xai_sdk.chat import user
                    chat.append(user(content))
                elif role == 'assistant':
                    from xai_sdk.chat import assistant
                    chat.append(assistant(content))
            
            # Sample the response (note: xai-sdk doesn't directly expose temperature/max_tokens in sample())
            # These parameters may need to be set differently depending on SDK version
            response = chat.sample()
            
            # Extract text from response
            response_text = response.content
            logger.debug(f"Grok response length: {len(response_text)} characters")
            
            # Create normalized response structure matching OpenAI format
            normalized = {
                'choices': [
                    {'message': {'content': response_text}}
                ],
                'model': model,
                'usage': {
                    'prompt_tokens': 0,  # xai-sdk doesn't expose token counts in the same way
                    'completion_tokens': 0,
                    'total_tokens': 0,
                }
            }
            
            return normalized
            
        except Exception as e:
            logger.error(f"Request failed: {e}", exc_info=True)
            raise
    
    def _call_anthropic_api(self, model: str, messages: List[Dict[str, Any]], max_tokens: int, **options) -> Dict[str, Any]:
        """Make API call to Anthropic Claude with specified model and parameters"""
        temperature = options.get('temperature', 0.1)
        timeout = options.get('timeout', 60)
        
        try:
            # Anthropic requires system message to be separate from messages array
            system_content = ""
            user_messages = []
            
            for msg in messages:
                role = msg.get('role', '')
                content = msg.get('content', '')
                
                if role == 'system':
                    system_content = content
                elif role == 'user':
                    user_messages.append({"role": "user", "content": content})
                elif role == 'assistant':
                    user_messages.append({"role": "assistant", "content": content})
            
            # Prepare base params
            params = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system_content if system_content else None,
                "messages": user_messages
            }
            
            # Add extra options (excluding internal ones)
            internal_options = {'temperature', 'timeout', 'headers'}
            for k, v in options.items():
                if k not in internal_options and k not in params:
                    params[k] = v

            # Call Anthropic API
            response = self.anthropic_client.messages.create(**params)
            
            # Extract text from response
            response_text = response.content[0].text
            logger.debug(f"Claude response length: {len(response_text)} characters")
            
            # Create normalized response structure matching OpenAI format
            normalized = {
                'choices': [
                    {'message': {'content': response_text}}
                ],
                'model': model,
                'usage': {
                    'prompt_tokens': response.usage.input_tokens,
                    'completion_tokens': response.usage.output_tokens,
                    'total_tokens': response.usage.input_tokens + response.usage.output_tokens,
                }
            }
            
            return normalized
            
        except Exception as e:
            logger.error(f"Anthropic request failed: {e}", exc_info=True)
            raise
    
    # ========================================================================
    # ASYNC VERSIONS OF API CALLS (Priority 6: Native Async Support)
    # ========================================================================
    
    async def _call_openai_api_async(self, model: str, messages: List[Dict[str, Any]], max_tokens: int, **options) -> Dict[str, Any]:
        """Async version: Make API call to OpenAI with specified model and parameters"""
        temperature = options.get('temperature', 0.1)
        timeout = options.get('timeout', 60)
        
        try:
            # Use AsyncOpenAI client
            response = await self.async_openai_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout
            )
            
            # Convert to normalized format
            return {
                'choices': [
                    {'message': {'content': response.choices[0].message.content}}
                ],
                'model': model,
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens if response.usage else 0,
                    'completion_tokens': response.usage.completion_tokens if response.usage else 0,
                    'total_tokens': response.usage.total_tokens if response.usage else 0,
                }
            }
        except Exception as e:
            logger.error(f"Async OpenAI request failed: {e}", exc_info=True)
            raise ApiConnectionError(f"OpenAI API failure: {str(e)}")
    
    async def _call_gemini_api_async(self, model: str, messages: List[Dict[str, Any]], max_tokens: int, **options) -> Dict[str, Any]:
        """Async version: Make API call to Google Gemini with specified model and parameters"""
        temperature = options.get('temperature', 0.1)
        
        try:
            # Combine messages into single prompt
            combined_prompt = "\n\n".join([
                f"{msg.get('role', '').upper()}: {msg.get('content', '')}"
                for msg in messages
            ])
            
            # Use Gemini async client
            from google import genai
            response = await self.genai_client.aio.models.generate_content(
                model=model,
                contents=combined_prompt,
                config=genai.types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                )
            )
            
            logger.debug(f"Gemini async response candidates: {len(response.candidates)}")
            response_text = response.text
            logger.debug(f"Response length: {len(response_text)} characters")
            
            # Create normalized response structure
            return {
                'choices': [
                    {'message': {'content': response_text}}
                ],
                'model': model,
                'usage': {
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'total_tokens': 0,
                }
            }
            
        except Exception as e:
            logger.error(f"Async Gemini request failed: {e}", exc_info=True)
            raise
    
    async def _call_grok_api_async(self, model: str, messages: List[Dict[str, Any]], max_tokens: int, **options) -> Dict[str, Any]:
        """Async version: Make API call to xAI Grok (using httpx since xai-sdk doesn't have async)"""
        temperature = options.get('temperature', 0.1)
        timeout = options.get('timeout', 60)
        
        try:
            # xai-sdk doesn't have async support, use httpx directly
            url = "https://api.x.ai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
            
            # Normalize response
            response_text = data["choices"][0]["message"]["content"]
            logger.debug(f"Grok async response length: {len(response_text)} characters")
            
            return {
                'choices': [
                    {'message': {'content': response_text}}
                ],
                'model': model,
                'usage': data.get('usage', {
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'total_tokens': 0,
                })
            }
            
        except Exception as e:
            logger.error(f"Async Grok request failed: {e}", exc_info=True)
            raise
    
    async def _call_anthropic_api_async(self, model: str, messages: List[Dict[str, Any]], max_tokens: int, **options) -> Dict[str, Any]:
        """Async version: Make API call to Anthropic Claude with specified model and parameters"""
        temperature = options.get('temperature', 0.1)
        timeout = options.get('timeout', 60)
        
        try:
            # Anthropic requires system message to be separate
            system_content = ""
            user_messages = []
            
            for msg in messages:
                role = msg.get('role', '')
                content = msg.get('content', '')
                
                if role == 'system':
                    system_content = content
                elif role == 'user':
                    user_messages.append({"role": "user", "content": content})
                elif role == 'assistant':
                    user_messages.append({"role": "assistant", "content": content})
            
            # Call async Anthropic API
            response = await self.async_anthropic_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_content if system_content else None,
                messages=user_messages,
                timeout=timeout
            )
            
            # Extract text from response
            response_text = response.content[0].text
            logger.debug(f"Claude async response length: {len(response_text)} characters")
            
            # Create normalized response structure
            return {
                'choices': [
                    {'message': {'content': response_text}}
                ],
                'model': model,
                'usage': {
                    'prompt_tokens': response.usage.input_tokens,
                    'completion_tokens': response.usage.output_tokens,
                    'total_tokens': response.usage.input_tokens + response.usage.output_tokens,
                }
            }
            
        except Exception as e:
            logger.error(f"Async Anthropic request failed: {e}", exc_info=True)
            raise
