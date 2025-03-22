import io
import os
import fitz
import time
import json
import openai
import logging
import autogen_agentchat.agents
from pdf2image import convert_from_bytes

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from awb.common import json_parsing_prompt, awb_schema, model_client_json_data_structuring_agent

client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

#Image OCR prompt
image_ocr_prompt_file=os.path.join("awb","prompts","image-ocr-prompt.txt")
with open(image_ocr_prompt_file, "r", encoding="utf-8") as image_ocr_prompt_data:
    image_ocr_prompt = image_ocr_prompt_data.read()  # Read the entire file as bytes

class LLMAgentParser:
    def __init__(self,filename:str,file_bytes):
        self.filename = filename
        self.file_bytes = file_bytes
        self.agent = self.create_agent()

    def create_agent(self):
        return autogen_agentchat.agents.AssistantAgent(
        name="awb_parser_llm_agent",
        model_client=model_client_json_data_structuring_agent,
        system_message= f"""
                            You are an expert that extracts Airway bill(AWB) from images and PDFs. 
                            You **MUST** use the function `execute_llm_based_ocr_agent` to process the file. 
                            Do not generate responses manually. Do not hallucinate or assume any missing information. 
                            If the tool returns empty, return None.
                            Do not lose any data provided in the input. Don't cut any data.   
                            You have deep understanding of the Airway bill (AWB) data structure.
                            
                            {json_parsing_prompt}                            
                        """,     
        tools=[self.execute_llm_based_ocr_agent],
        model_client_stream=True,
        reflect_on_tool_use=True,

    )

    def execute_llm_based_ocr_agent(self):
        '''
        LLM Based OCR tool to extract the data from files and images
        '''
        logging.info("Using LLM-Based OCR Tool.---------------")
        logging.info("Waiting for processing at the start...")
    
        structured_response = None  # Initialize to None
        file_id = ""

        try:

            file_id, is_pdf = self.process_file(self.file_bytes,self.filename)
            print('*************************',self.filename,"***",is_pdf)
            
            # create an openai assistant
            assistant = openai.beta.assistants.create(
                name="AWB parser and processor",
                model="gpt-4o",
                instructions=f"""
                            # #You are an OCR agent that extracts text from images and PDFs. "
                            
                            # You are an expert who extracts Airway bill(AWB) text data from images and PDFs and other
                            uploaded files.  {image_ocr_prompt}      
                                
                            "# Do not lose any data provided in the input. Don't cut any data"
                            " # You have a deep understanding of Airway bills data structure." 
                            {json_parsing_prompt}         
                                                
                        """,               
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "awb_schema",
                        "strict": True,
                        "schema": json.loads(awb_schema)
                    }
                },
                tools=[{"type": "file_search"}],
            )

            # ✅ Create Thread
            thread = client.beta.threads.create(
                messages=[
                    {
                        "role": "user",
                        "content": "Extract the airway bill (AWB)  data from the file.",
                    }
                ]
            )
            
            thread_id = thread.id
            print("thread id -------- " + thread_id)
            
            content = [
                {"type": "text", "text": "Extract Airway bill(AWB) data from uploaded file"},
            ]

            tools_object = [{"type": "file_search"}] if is_pdf else None
            attachments = [{"file_id": file_id, "tools": tools_object}] if is_pdf else None
            content = [{"type": "image_file", "image_file": {"file_id": file_id, "detail": "high"}}] if not is_pdf else None
            print("*****************",is_pdf)
            if is_pdf:
                openai.beta.threads.messages.create(
                    thread_id=thread_id,
                    role="user",
                    attachments=attachments,
                    content= "Extract Airway bill(AWB) data from uploaded file",
                )
            else:
                openai.beta.threads.messages.create(
                    thread_id=thread_id,
                    role="user",
                    content = content
                )
            

            # ✅ run assistant in thread
            run = openai.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=assistant.id,
                instructions="Return structured JSON output.",
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "awb_schema",
                        "strict": True,
                        "schema": json.loads(awb_schema)
                    }
                }  # 🔹 Enforce JSON output
            )

            # wait for completion
            while True:
                run_status = openai.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
                if run_status.status == "completed":
                    break
                print("Waiting for processing...")
                time.sleep(2)

            # fetch message and parse to json
            structured_response = None
            messages = openai.beta.threads.messages.list(thread_id=thread_id)        
            for msg in messages.data:
                if msg.role == "assistant" and msg.content and msg.content[0].type == "text":
                    try:
                        structured_response = json.loads(msg.content[0].text.value)
                        print("Structured JSON Response:", json.dumps(structured_response, indent=2))
                        break
                    except Exception as ef:
                        print(f"Error: Assistant response is not valid JSON.: {ef}")
        
            self.delete_file_from_openai(file_id)
            return structured_response if structured_response else None

        except Exception as e:
            print(f"An error occurred: {e}")  # Added print statement.
            self.delete_file_from_openai(file_id)
            return None   

    def create_inmemory_file_object_from_largest(self,images):
        """Creates a binaryIO object for the largest image in a PDF."""
        largest_image = max(images, key=lambda img: img.width * img.height, default=None)
        if largest_image:
            buffered = io.BytesIO()
            largest_image.save(buffered, format="JPEG")
            buffered.seek(0)
            return buffered
        return None

    def upload_single_image_file_to_openAI(self,file_bytes):
        """Uploads a single image file to OpenAI."""
        if file_bytes:
            file_bytes.name = "image.jpg"
            uploaded_file = openai.files.create(file=file_bytes, purpose="vision")
            return uploaded_file.id
        return None

    def upload_file_to_openAI(self,file_bytes):
        """Uploads a PDF file to OpenAI."""
        if file_bytes:
            file_bytes.name = "document.pdf"
            uploaded_file = openai.files.create(file=file_bytes, purpose="assistants")
            return uploaded_file.id
        return None

    def has_image_dominant_page(self,pdf_document, page_number):
        """Checks if a specific PDF page has more image data than text."""
        try:
            # pdf_document = fitz.open(pdf_path)
            page = pdf_document[page_number]

            text_length = len(page.get_text("text"))
            image_list = page.get_images(full=True)
            total_image_size = sum(len(fitz.Pixmap(pdf_document, img[0]).tobytes()) for img in image_list)

            pdf_document.close()

            return total_image_size > text_length

        except Exception as e:
            print(f"Error processing PDF: {e}")
            return False

    def process_file(self,file_bytes,filename:str):
        """
        Determines file type and processes it accordingly:
        - If image, uploads directly.
        - If PDF, checks for dominant images and uploads accordingly.
        
        Returns:
            tuple: (file_id, is_pdf)
        """
        file_id = ""
        is_pdf = True  # Default to True; will change to False if an image is found.

        file_extension = os.path.splitext(filename)[-1].lower()

        # If the file is an image (jpg, jpeg, png), upload it directly
        if file_extension in [".jpg", ".jpeg", ".png"]:
            print(f"Processing image file: {filename}")
            image_io = io.BytesIO(file_bytes)
            file_id = self.upload_single_image_file_to_openAI(image_io)
            is_pdf = False  # It's an image, not a pure PDF
            return file_id, is_pdf

        # If the file is a PDF, process accordingly
        elif file_extension == ".pdf":
            print(f"Processing PDF file: {filename}")

            pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
            image_dominant_page = None

            # Check each page for image dominance
            for page_number in range(len(pdf_document)):
                if self.has_image_dominant_page(pdf_document, page_number):
                    image_dominant_page = page_number
                    is_pdf = False  # Found at least one image page
                    break  # Stop after finding the first image-dominant page

            if image_dominant_page is not None:
                print(f"Page {image_dominant_page + 1} has a dominant image. Extracting...")
                images = convert_from_bytes(file_bytes, first_page=image_dominant_page + 1, last_page=image_dominant_page + 1)
                if images:
                    awb_image_io = self.create_inmemory_file_object_from_largest(images)
                    if awb_image_io:
                        file_id = self.upload_single_image_file_to_openAI(awb_image_io)
                        return file_id, is_pdf

            print(f"No dominant images found in PDF. Uploading entire PDF.")
            pdf_io = io.BytesIO(file_bytes)
            file_id = self.upload_file_to_openAI(pdf_io)
            return file_id, is_pdf

        else:
            print(f"Unsupported file type: {file_extension}")
            return None, None

    def delete_file_from_openai(self,file_id_to_delete: str):
        """
        Deletes the uploaded file from openAI
        """
        try:
            response = openai.files.delete(file_id_to_delete)
            if response.deleted:
                print(f"File '{file_id_to_delete}' deleted successfully fro OpenAI.")
            else:
                print(f"Failed to delete file '{file_id_to_delete}'.")
        except openai.OpenAIError as e:
            print(f"An OpenAIError error occurred while deleting the file: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while deleting fil from openAI: {e}")
    