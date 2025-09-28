"""
Decorator utilities for the Bitcoin trading bot.
"""

import time
import functools
from typing import Any, Callable, Optional
from trading_bot.utils.logger import get_logger

logger = get_logger(__name__)


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Retry decorator with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier for delay

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            current_delay = delay
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}: {e}"
                        )

            raise last_exception

        return wrapper
    return decorator


def timing(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.

    Args:
        func: Function to time

    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            execution_time = end_time - start_time
            logger.debug(f"{func.__name__} executed in {execution_time:.3f}s")

    return wrapper


def rate_limit(calls_per_second: float):
    """
    Rate limiting decorator.

    Args:
        calls_per_second: Maximum calls per second allowed

    Returns:
        Decorated function
    """
    min_interval = 1.0 / calls_per_second
    last_called = [0.0]

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed

            if left_to_wait > 0:
                time.sleep(left_to_wait)

            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret

        return wrapper
    return decorator


def cache_result(ttl: Optional[float] = None):
    """
    Simple caching decorator with optional TTL.

    Args:
        ttl: Time to live in seconds (None for no expiration)

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        cache = {}

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Create cache key from arguments
            key = str(args) + str(sorted(kwargs.items()))

            # Check if cached result exists and is still valid
            if key in cache:
                result, timestamp = cache[key]
                if ttl is None or time.time() - timestamp < ttl:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return result
                else:
                    # Remove expired entry
                    del cache[key]

            # Call function and cache result
            result = func(*args, **kwargs)
            cache[key] = (result, time.time())
            logger.debug(f"Cache miss for {func.__name__}, result cached")

            return result

        # Add cache management methods
        wrapper.cache_clear = lambda: cache.clear()
        wrapper.cache_info = lambda: {
            'size': len(cache),
            'keys': list(cache.keys())
        }

        return wrapper
    return decorator


def validate_inputs(**validators):
    """
    Decorator to validate function inputs.

    Args:
        **validators: Validation functions for each parameter

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Validate each parameter
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if not validator(value):
                        raise ValueError(
                            f"Invalid value for parameter '{param_name}': {value}"
                        )

            return func(*args, **kwargs)

        return wrapper
    return decorator


def error_handler(default_return: Any = None, log_error: bool = True):
    """
    Decorator to handle errors gracefully.

    Args:
        default_return: Default value to return on error
        log_error: Whether to log the error

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    logger.error(f"Error in {func.__name__}: {e}")
                return default_return

        return wrapper
    return decorator