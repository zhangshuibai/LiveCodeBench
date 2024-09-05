import os
from time import sleep

try:
    import openai
    from openai import OpenAI
except ImportError as e:
    pass

from lcb_runner.runner.base_runner import BaseRunner


class gpt_agent():
    def __init__(self, role_description, model_name):
        self.client = OpenAI(
        api_key=os.getenv("OPENAI_KEY"),
    )
        self.client_kwargs: dict[str | str] = {
            "model": model_name,
            "temperature": 0.2,
            "max_tokens": 8192,
            "top_p": 0.95,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "n": 1,
            "timeout": 45,
            # "stop": args.stop, --> stop is only used for base models currently
        }

    def __call__(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                messages=prompt,
                **self.client_kwargs,
            )
        except (
            openai.APIError,
            openai.RateLimitError,
            openai.InternalServerError,
            openai.OpenAIError,
            openai.APIStatusError,
            openai.APITimeoutError,
            openai.InternalServerError,
            openai.APIConnectionError,
        ) as e:
            print("Exception: ", repr(e))
            print("Sleeping for 30 seconds...")
            print("Consider reducing the number of parallel processes.")
            sleep(30)
            return self.__call__(prompt)
        except Exception as e:
            print(f"Failed to run the model for {prompt}!")
            print("Exception: ", repr(e))
            raise e
        return [c.message.content for c in response.choices]

        # pass
    def _run_single(self, prompt: list[dict[str, str]]) -> list[str]:
        assert isinstance(prompt, list)

        try:
            response = self.client.chat.completions.create(
                messages=prompt,
                **self.client_kwargs,
            )
        except (
            openai.APIError,
            openai.RateLimitError,
            openai.InternalServerError,
            openai.OpenAIError,
            openai.APIStatusError,
            openai.APITimeoutError,
            openai.InternalServerError,
            openai.APIConnectionError,
        ) as e:
            print("Exception: ", repr(e))
            print("Sleeping for 30 seconds...")
            print("Consider reducing the number of parallel processes.")
            sleep(30)
            return OpenAIRunner._run_single(prompt)
        except Exception as e:
            print(f"Failed to run the model for {prompt}!")
            print("Exception: ", repr(e))
            raise e
        return [c.message.content for c in response.choices]
