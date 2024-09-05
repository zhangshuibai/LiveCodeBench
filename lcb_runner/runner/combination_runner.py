import os
from time import sleep

try:
    import openai
    from openai import OpenAI
except ImportError as e:
    pass

try:
    from mistralai.client import MistralClient
except ImportError as e:
    pass

try:
    from anthropic import Anthropic
except ImportError as e:
    pass


from lcb_runner.runner.base_runner import BaseRunner


class combination_code_generator:
    def __init__(self) -> None:
        self.max_test_times = 3
        self.ANALYST_model = "deepseek-chat"#"gpt-3.5-turbo"
        self.DEVELOPER_model = "deepseek-chat"#"gpt-3.5-turbo"
        self.REPAIR_model = "deepseek-chat"#"gpt-3.5-turbo"
        self.TESTER_model = "deepseek-chat"#"gpt-3.5-turbo"

        self.ANALYST_description = '''I want you to act as a requirement analyst on our development team. Given a user requirement, your task is to analyze, decompose, and develop a high-level plan to guide our developer in writing programs. The plan should include the following information:
1. Decompose the requirement into several easy-to-solve subproblems that can be more easily implemented by the developer.
2. Develop a high-level plan that outlines the major steps of the program.
Remember, your plan should be high-level and focused on guiding the developer in writing code, rather than providing implementation details.
'''

        self.DEVELOPER_description = '''Write codes that meets the requirements following the plan. Ensure that the code you write is efficient, readable, and follows best practices. Remember, do not need to explain the code you wrote.
'''
        self.REPAIR_description = '''Fix or improve the code based on the content of the report. Ensure that any changes made to the code do not introduce new bugs or negatively impact the performance of the code. Remember, do not need to explain the code you wrote.
'''
        self.TESTER_description = '''I want you to act as a tester in the team. You will receive the code written by the developer, and your job is to complete a report as follows:
{
"Code Review": Evaluate the structure and syntax of the code to ensure that it conforms to the specifications of the programming language, that the APIs used are correct, and that the code does not contain syntax errors or logic holes.
"Code Description": Briefly describe what the code is supposed to do. This helps identify differences between the code implementation and the requirement.
"Satisfying the requirements": Ture or False. This indicates whether the code satisfies the requirement.
"Edge cases": Edge cases are scenarios where the code might not behave as expected or where inputs are at the extreme ends of what the code should handle.
"Conclusion": "Code Test Passed" or "Code Test Failed". This is a summary of the test results.
}
'''
        
        self.TEAM_description = '''There is a development team that includes a requirements analyst, a developer, a quality assurance tester and a repair that refine the code based on the report given by tester. The team needs to develop programs that satisfy the requirements of the users. The different roles have different divisions of labor and need to cooperate with each others.
        '''

        self.developer_instruction = "You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.\n"     

        self.ANALYST_agent = self._initialize_agent(role_description = self.TEAM_description + self.ANALYST_description, model_name = self.ANALYST_model)
        self.DEVELOPER_agent = self._initialize_agent(role_description = self.TEAM_description + self.developer_instruction + self.DEVELOPER_description, model_name = self.DEVELOPER_model)
        self.REPAIR_agent = self._initialize_agent(role_description = self.TEAM_description + self.REPAIR_description ,model_name = self.REPAIR_model)
        self.TESTER_agent = self._initialize_agent(role_description = self.TEAM_description + self.TESTER_description, model_name = self.TESTER_model)


    def _initialize_agent(self, role_description: str, model_name: str) -> callable:
        if model_name == "gpt-3.5-turbo" or model_name == "gpt-4-turbo" or model_name == "gpt-4o":
            raise NotImplementedError
        #     from api_model_single_call import gpt_agnent
        #     return gpt_agnent(role_description = role_description, model_name = "gpt-3.5-turbo")
        # elif model_name == "mistral":
        #     from api_model_single_call import mistral_agent
        #     return mistral_agent(role_description = role_description)
        # elif model_name == "gemini":
        #     from api_model_single_call import gemini_agent
        #     return gemini_agent(role_description = role_description)
        elif model_name == "deepseek-chat":
            from lcb_runner.runner.api_model_single_call.deepseek_agent import deepseek_agent
            return deepseek_agent(role_description = role_description, model_name=model_name)
        # elif model_name == "cohere":
        #     from api_model_single_call import cohere_agent
        #     return cohere_agent(role_description = role_description)
        # elif model_name == "claude":
        #     from api_model_single_call import claude_agent
        #     return claude_agent(role_description = role_description)
        else:
            raise NotImplementedError
        
    def forward(self, prompt: str) -> str:
        requirements_plan = self.ANALYST_agent(f"The requirement is {prompt}\n")
        code = self.DEVELOPER_agent(f"The plan from the requirement analyst is as following:\n{requirements_plan}\n")
        for i in range(self.max_test_times):
            test_report = self.TESTER_agent(f"The requirement is {prompt}\n" + f"The code provided by developer is as follows:\n{code}\n")
            if "Code Test Passed" in test_report:
                return code
            else:
                code = self.REPAIR_agent(f"The requirement is {prompt}\n" +
                                         f"The code provided by developer is as follows:\n{code}\n" + 
                                         f"The report from the tester is as following:\n{test_report}")
        
        return code


        
        



class CombinationRunner(BaseRunner):

    client = combination_code_generator()

    def __init__(self, args, model):
        super().__init__(args, model)
        

    def _run_single(self, prompt: str) -> list[str]:
        # assert isinstance(prompt, str)

        def __run_single(counter):
            try:
                content = self.client.forward(prompt=prompt)
                return content
            except Exception as e:
                print(f"Failed to run the model for {prompt}!")
                print("Exception: ", repr(e))
                raise e

        outputs = []
        try:
            for _ in range(self.args.n):
                outputs.append(__run_single(10))
        except Exception as e:
            raise e
        return outputs