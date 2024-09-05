import os
from time import sleep

try:
    import openai
    from openai import OpenAI
except ImportError as e:
    pass

# from lcb_runner.runner.base_runner import BaseRunner


class deepseek_agent:
    def __init__(self, role_description="", model_name=""):
        os.environ["DEEPSEEK_API"] = "sk-647096d358d04461b57e17f5c72345a4"
        self.client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API"), 
        # base_url="https://api.deepseek.com/beta",
        base_url="https://api.deepseek.com"
        )
        self.system_message = {
                "role": "system",
                "content": role_description,
            }
        
        
        self.client_kwargs: dict[str | str] = {
            "model": model_name,
            "temperature": 0.2,
            "max_tokens": 4096,
            "top_p": 0.95,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "n": 1,
            "timeout": 180,
            # "stop": args.stop, --> stop is only used for base models currently
        }

    def __call__(self, prompt: str) -> str:
        chat_messages = []
        chat_messages.append(self.system_message)
        chat_messages.append({
                "role": "user",
                "content": prompt,})

        try:
            response = self.client.chat.completions.create(
                messages=chat_messages,
                **self.client_kwargs,
            )
            content = response.choices[0].message.content
            # print(content)
            return content
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

       