import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key="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"
,model_name="llama-3.1-70b-versatile")

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the 
            following keys: `role`, `experience`, `skills`, and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):    
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]

    def write_mail(self, job, links):
        prompt_email = PromptTemplate.from_template(
             """
            ### JOB DESCRIPTION:
            {job_description}
            
            ### INSTRUCTION:
           You are Sandeep Kumar, a highly motivated and skilled Computer Science graduate seeking
           an opportunity to contribute to a dynamic and innovative organization. With a strong foundation in
           Software engineering, Python, SQL, DSA, Power BI and Problem Solving skills, combined
           with hands-on project experience, you are confident in your ability to deliver meaning results.
           
           You have completed some internships in the field of Data Sciecne and Machine Learning and 
           empowered the organization with hightend overall effieciency, Cost reduction and work optimization.
           
           
           Your job is to write a cold email to the client regarding the job mentioned above describing the capability of your skills 
            in fulfilling their needs.
            Also add the most relevant ones from the following links to showcase Yours portfolio: {link_list}
            Remember you are Sandeep,an enthusiast student to learn and grow. 
            Do not provide a preamble.
            ### EMAIL (NO PREAMBLE):
            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return res.content

if __name__ == "__main__":
    print("API Key Fetched")