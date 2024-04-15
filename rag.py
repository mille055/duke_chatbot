import argparse
import hashlib
import io, re, os
import json
import numpy as np
#from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, PipelineTool
from openai import OpenAI
from pinecone import init, Index, Pinecone, ServerlessSpec
from dotenv import load_dotenv
import logging
import requests
#import texthero as hero

# Load environment variables from .env file
load_dotenv()

class RAG:
    def __init__(self, openai_embedding_model = 'text-embedding-3-small', openai_engine='gpt-3.5-turbo', top_k=3, search_threshold=0.8, max_token_length=512, chunk_size = 500, chunk_overlap = 25, pinecone_index_name = None, llm_url = None, use_gpt = False, verbose=False):
        #pinecone
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')     
        # index: can pass index or get environment variable; if none, use default
        if pinecone_index_name:
            self.pinecone_index_name = pinecone_index_name
        else:
            self.pinecone_index_name = os.getenv('PINECONE_INDEX_NAME', 'dukechatbot0413')

        #openai
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.openai_embedding_model = openai_embedding_model
        self.openai_engine = openai_engine
        self.openai_client =  OpenAI(api_key=self.openai_api_key)
        self.use_gpt = use_gpt
        #text chunking and semantic search
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.search_threshold = search_threshold
        self.max_token_length = max_token_length
        self.prompt_instruction = prompt_instruction = "Please provide a concise summary to prospective students who are seeking answers to questions. Please generate text in complete sentences. Here is some context and the question: "
        self.verbose = verbose

        # our model 
        if llm_url:
            self.llm_url = llm_url
        else:
            self.llm_url = os.getenv('LLM_URL', 'https://ej0lhmgikhbq6zp9.us-east-1.aws.endpoints.huggingface.cloud')
        self.llm_token = os.getenv('HUGGINGFACE_TOKEN')


        self.text_to_replace = ["© Copyright 2011-2024 Duke University * __Main Menu * __Why Duke? * __The Duke Difference * __Career Services * __Graduate Outcomes * __What Tech Leaders Are Saying * __Degree * __Courses * __Faculty * __Advisory Board * __Apply * __Quick Links * __News * __Events * __Steering Committee * __Contact *",
                   "Jump to navigation * Duke Engineering * Pratt School of Engineering * Institute for Enterprise Engineering * News * Events * Steering Committee * Contact __ * Why Duke? * The Duke Difference * Career Services * Graduate Outcomes * What Tech Leaders Are Saying * Degree * Courses * Faculty * Advisory Board * Apply __",
                   "Skip to main content * Departments & Centers * Overview * Biomedical Engineering * Civil & Environmental Engineering * Electrical & Computer Engineering * Mechanical Engineering & Materials Science * Institute for Enterprise Engineering * Alumni & Parents * Overview * Alumni * Parents * Giving * Board of Visitors * Our History * Email Newsletter * Meet the Team * Corporate Partners * Overview * Partners & Sponsors * Data Science & AI Industry Affiliates * Connect With Students * Recruiting Our Students * Sponsored Research * TechConnect Career Networking * Apply * Careers * Directory * Undergraduate * 1. For Prospective Students 1. Majors & Minors 2. Certificates 3. General Degree Requirements 4. 4+1: BSE+Master's Degree 5. Campus Tours 6. How to Apply 2. First-Year Design 3. Student Entrepreneurship 4. Undergraduate Research 5. Where Our Undergrads Go 6. Diversity, Equity & Inclusion 7. For Current Students 1. The First Year 2. Advising 3. Student Clubs & Teams 4. Graduation with Distinction 5. Internships 6. Policies & Procedures * Graduate * 1. For Prospective Students 1. PhD Programs 2. Master's Degrees 3. Online Specializations, Certificates and Short Courses 4. Admissions Events 5. How to Apply 2. For Admitted Students 3. Diversity, Equity & Inclusion 1. Bootcamp for Applicants 2. Recruiting Incentives 4. For Current Grad Students 1. Graduate Student Programs & Services * Faculty & Research * 1. Faculty 1. Faculty Profiles 2. New Faculty 3. Awards and Recognition 4. NAE Members 2. Research 1. Signature Research Themes 2. Recent External Funding Awards 3. Faculty Entrepreneurship 4. Duke Engineering Discoveries * About * 1. Dean's Welcome 2. Campus & Tours 3. Facts & Rankings 4. Diversity, Equity & Inclusion 5. Service to Society 6. Entrepreneurship 7. Governance 8. News & Media 1. Latest News 2. Podcast 3. Email Newsletter 4. Publications 5. Media Coverage 6. Public Health Information 9. Events 1. Events Calendar 2. Academic Calendar 3. Commencement 10. Art @ Duke Engineering",
                   "Skip to main content * Duke University » * Pratt School of Engineering » ## Secondary Menu * Apply * Careers * Contact * Undergraduate * 1. Admissions 1. Degree Program 2. Enrollment and Graduation Rates 3. Career Outcomes 4. Campus Tours 5. How to Apply 2. Academics 1. Curriculum 2. Double Majors 3. BME Design Fellows 3. Student Resources 1. For Current Students 2. 4+1: BSE+Master's Degree * Master's * 1. Admissions 1. Degree Programs 2. Career Outcomes 3. How to Apply 2. Academics 1. Courses 2. Concentrations 3. Certificates 3. Student Resources 1. For Current Students * PhD * 1. Admissions 1. PhD Program 2. Meet Our Students 3. Career Outcomes 4. How to Apply 2. Academics 1. Courses 2. Certificates & Training Programs 3. Student Resources 1. For Current Students * Research * 1. Major Research Programs 2. Centers & Initiatives 3. Research News * Faculty * 1. Faculty Profiles 2. Awards & Recognition * Coulter * 1. The Duke-Coulter Partnership 2. Proposal Process 3. Project Archive 4. Oversight Committee 5. FAQs * About * 1. Welcome from the Chair 2. Vision & Mission 3. Facts & Stats 4. Serving Society 5. News 1. Media Coverage 2. Duke BME Magazine 3. Email Newsletter 6. Events 1. Seminars 7. Our History 8. Driving Directions",
                   "Jump to navigation * Duke Engineering * Pratt School of Engineering * Institute for Enterprise Engineering * Industry Relations * Leadership * News * Contact __ * Why Duke? * The Duke Difference * Career Services * Graduate Outcomes * What Tech Leaders Are Saying * Degree * Certificate * Courses * Faculty * Apply",
                   "# Page Not Found We're sorry, but that page was not found. Please check the spelling of the page address or search this website. * * * * * © Copyright 2011-2024 Duke University drupal_block( 'search_form_block', { label_display: false } ) * Undergraduate * Admissions * Degree Program * Enrollment and Graduation Rates * Career Outcomes * Campus Tours * How to Apply * Academics * Curriculum * Double Majors * BME Design Fellows * Student Resources * For Current Students * 4+1: BSE+Master's Degree * Master's * Admissions * Degree Programs * Career Outcomes * How to Apply * Academics * Courses * Concentrations * Certificates * Student Resources * For Current Students * PhD * Admissions * PhD Program * Meet Our Students * Career Outcomes * How to Apply * Academics * Courses * Certificates & Training Programs * Student Resources * For Current Students * Research * Major Research Programs * Centers & Initiatives * Research News * Faculty * Faculty Profiles * Awards & Recognition * Coulter * The Duke-Coulter Partnership * Proposal Process * Project Archive * Oversight Committee * FAQs * About * Welcome from the Chair * Vision & Mission * Facts & Stats * Serving Society * News * Media Coverage * Duke BME Magazine * Email Newsletter * Events * Seminars * Past Seminars * Our History * Driving Directions",
                   "Skip to main content * Departments & Centers * Overview * Biomedical Engineering * Civil & Environmental Engineering * Electrical & Computer Engineering * Mechanical Engineering & Materials Science * Institute for Enterprise Engineering * Alumni & Parents * Overview * Alumni * Parents * Giving * Board of Visitors * Our History * Email Newsletter * Meet the Team * Corporate Partners * Overview * Partners & Sponsors * Data Science & AI Industry Affiliates * Connect With Students * Recruiting Our Students * Sponsored Research * TechConnect Career Networking * Apply * Careers * Directory * Undergraduate * 1. For Prospective Students 1. Majors & Minors 2. Certificates 3. General Degree Requirements 4. 4+1: BSE+Master's Degree 5. Campus Tours 6. How to Apply 2. First-Year Design 3. Student Entrepreneurship 4. Undergraduate Research 5. Where Our Undergrads Go 6. Diversity, Equity & Inclusion 7. For Current Students 1. The First Year 2. Advising 3. Student Clubs & Teams 4. Graduation with Distinction 5. Internships 6. Policies & Procedures * Graduate * 1. For Prospective Students 1. PhD Programs 2. Master's Degrees 3. Online Specializations, Certificates and Short Courses 4. Admissions Events 5. How to Apply 2. For Admitted Students 3. Diversity, Equity & Inclusion 1. Bootcamp for Applicants 2. Recruiting Incentives 4. For Current Grad Students 1. Graduate Student Programs & Services * Faculty & Research * 1. Faculty 1. Faculty Profiles 2. New Faculty 3. Awards and Recognition 4. NAE Members 2. Research 1. Signature Research Themes 2. Recent External Funding Awards 3. Faculty Entrepreneurship 4. Duke Engineering Discoveries * About * 1. Dean's Welcome 2. Campus & Tours 3. Facts & Rankings 4. Diversity, Equity & Inclusion 5. Service to Society 6. Entrepreneurship 7. Governance 8. News & Media 1. Latest News 2. Podcast 3. Email Newsletter 4. Publications 5. Media Coverage 6. Public Health Information 9. Events 1. Events Calendar 2. Academic Calendar 3. Commencement 10. Art @ Duke Engineering ## You are here Home » About » News & Media",
                   "Skip to main content * Duke University » * Pratt School of Engineering » ## Secondary Menu * Apply * Careers * Contact * Undergraduate * 1. The Duke Difference 1. Why Duke for CEE? 2. Enrollment and Graduation Rates 3. Where Our Students Go 2. Degree Options 1. Civil Eng Degree Planning 2. Civil Eng Study Tracks 3. Env. Eng Degree Planning 4. Dual Majors 5. Certificates 6. 4+1: BSE+Master's 3. For Current Students 1. Courses 2. Research & Independent Study 3. Outreach & Service Learning 4. Senior Design Capstone 5. Internships & Career Planning 6. Graduation with Distinction * Graduate * 1. The Duke Difference 1. Degree Options 2. Scholarships & Financial Support 3. Graduate Study Tracks 4. Graduate Certificates 5. Course Descriptions 2. Master's Study 1. Master of Science in CEE 2. Civil Engineering 3. Computational Mechanics and Scientfic Computing 4. Environmental Engineering 5. Risk Engineering 6. Career Services 7. Career Outcomes 3. PhD 1. Meet Our Students 2. PhD Career Outcomes 4. For Current Students * Research * 1. Overview 2. Research News 3. Computational Mechanics & Scientific Computing 4. Environmental Health Engineering 5. Geomechanics & Geophysics for Energy and the Environment 6. Hydrology & Fluid Dynamics 7. Risk & Resilient Systems * Faculty * 1. Faculty Profiles 2. Awards & Recognition * About * 1. Facts & Stats 2. Meet the Chair 3. Serving a Global Society 4. Advisory Board 5. Alumni 6. Awards & Honors 7. News 1. Media Coverage 2. Email Newsletter 3. Duke CEE Magazine 8. Events 1. Seminars",
                   "# Page Not Found We're sorry, but that page was not found. Please check the spelling of the page address or search this website. * * * * * © Copyright 2011-2024 Duke University drupal_block( 'search_form_block', { label_display: false } ) * Undergraduate * The Duke Difference * Why Duke for CEE? * Enrollment and Graduation Rates * Where Our Students Go * Degree Options * Civil Eng Degree Planning * Civil Eng Study Tracks * Env. Eng Degree Planning * Dual Majors * Certificates * Architectural Engineering * Global Development * Energy & The Environment * 4+1: BSE+Master's * For Current Students * Courses * Research & Independent Study * Outreach & Service Learning * Senior Design Capstone * Internships & Career Planning * Graduation with Distinction * Graduate * The Duke Difference * Degree Options * Scholarships & Financial Support * Graduate Study Tracks * Graduate Certificates * Course Descriptions * Master's Study * Master of Science in CEE * Civil Engineering * Computational Mechanics and Scientfic Computing * Environmental Engineering * Risk Engineering * Career Services * Career Outcomes * PhD * Meet Our Students * PhD Career Outcomes * For Current Students * Research * Overview * Research News * Computational Mechanics & Scientific Computing * Environmental Health Engineering * Geomechanics & Geophysics for Energy and the Environment * Hydrology & Fluid Dynamics * Risk & Resilient Systems * Faculty * Faculty Profiles * Awards & Recognition * About * Facts & Stats * Meet the Chair * Serving a Global Society * Advisory Board * Alumni * Awards & Honors * News * Media Coverage * Email Newsletter * Duke CEE Magazine * Events * Seminars * Past Seminars",
                   "Copyright 2011-2024 Duke University drupal_block( 'search_form_block', { label_display: false } ) * Undergraduate * The Duke Difference * Why Duke for CEE? * Enrollment and Graduation Rates * Where Our Students Go * Degree Options * Civil Eng Degree Planning * Civil Eng Study Tracks * Env. Eng Degree Planning * Dual Majors * Certificates * Architectural Engineering * Global Development * Energy & The Environment * 4+1: BSE+Master's * For Current Students * Courses * Research & Independent Study * Outreach & Service Learning * Senior Design Capstone * Internships & Career Planning * Graduation with Distinction * Graduate * The Duke Difference * Degree Options * Scholarships & Financial Support * Graduate Study Tracks * Graduate Certificates * Course Descriptions * Master's Study * Master of Science in CEE * Civil Engineering * Computational Mechanics and Scientfic Computing * Environmental Engineering * Risk Engineering * Career Services * Career Outcomes * PhD * Meet Our Students * PhD Career Outcomes * For Current Students * Research * Overview * Research News * Computational Mechanics & Scientific Computing * Environmental Health Engineering * Geomechanics & Geophysics for Energy and the Environment * Hydrology & Fluid Dynamics * Risk & Resilient Systems * Faculty * Faculty Profiles * Awards & Recognition * About * Facts & Stats * Meet the Chair * Serving a Global Society * Advisory Board * Alumni * Awards & Honors * News * Media Coverage * Email Newsletter * Duke CEE Magazine * Events * Seminars * Past Seminars",
                   "Page Not Found We're sorry, but that page was not found. Please check the spelling of the page address or search this website. * * * * * © Copyright 2011-2024 Duke University drupal_block( 'search_form_block', { label_display: false } ) * Undergraduate * Overview * Degree Programs * BSE Degree Planning * Areas of Concentration * Concentration in Machine Learning * Minor in ECE * Minor in Machine Learning & AI * For Applicants * Enrollment and Graduation Rates * Where Our Students Go * What's the difference between CS and ECE? * For Current Students * Courses * Innovations in Remote Learning * Independent Study * Senior Design * Graduation with Distinction * Awards and Honors * Research Experiences for Undergrads (REU) * Master's * Overview * Degree Options * Master of Science (MS) * Master of Engineering (MEng) * Areas of Study * Software Development * Hardware Design * Data Analytics & Machine Learning * Quantum Computing * Microelectronics, Photonics & Nanotechnology * Design Your Own ECE Degree * Master's Admissions * Master's Career Outcomes * Life at Duke * Research Opportunities * Graduate Courses * Online Courses * PhD * Overview * Degree Requirements * Academic Curricular Groups * PhD Admissions * Promoting an Inclusive Environment * Meet Our Students * PhD Awards and Honors * PhD Career Outcomes * Certificates & Training Programs * Graduate Courses * DEEP SEA Startup Accelerator * Career & Professional Services * Faculty & Research * Overview * AI/Machine Learning * Metamaterials * Quantum Computing * Nanoelectronic Materials & Devices * Sensing & Imaging * Trustworthy Computing * Faculty Profiles * Awards & Recognition * Research News * Ask an Expert * About * From the Chair * News * Media Coverage * Email Newsletter * Duke ECE Magazine * Events * Distinguished Speaker Series * Seminars * Past Seminars * Facts & Stats * Mission & Vision * Diversity, Equity, Inclusion & Community * Entrepreneurship Success Stories * Meet Our Alumni * Industry Advisory Board",
                   "Skip to main content * Duke University » * Pratt School of Engineering » ## Secondary Menu * Apply * Careers * Contact * Undergraduate * 1. Overview 2. Degree Programs 1. BSE Degree Planning 2. Areas of Concentration 3. Concentration in Machine Learning 4. Minor in ECE 5. Minor in Machine Learning & AI 3. For Applicants 1. Enrollment and Graduation Rates 2. Where Our Students Go 3. What's the difference between CS and ECE? 4. For Current Students 1. Courses 2. Innovations in Remote Learning 3. Independent Study 4. Senior Design 5. Graduation with Distinction 6. Awards and Honors 5. Research Experiences for Undergrads (REU) * Master's * 1. Overview 2. Degree Options 1. Master of Science (MS) 2. Master of Engineering (MEng) 3. Areas of Study 1. Software Development 2. Hardware Design 3. Data Analytics & Machine Learning 4. Quantum Computing 5. Microelectronics, Photonics & Nanotechnology 6. Design Your Own ECE Degree 4. Master's Admissions 5. Master's Career Outcomes 6. Life at Duke 7. Research Opportunities 8. Graduate Courses 9. Online Courses * PhD * 1. Overview 2. Degree Requirements 3. Academic Curricular Groups 4. PhD Admissions 5. Promoting an Inclusive Environment 6. Meet Our Students 1. PhD Awards and Honors 7. PhD Career Outcomes 8. Certificates & Training Programs 9. Graduate Courses 10. DEEP SEA Startup Accelerator 11. Career & Professional Services * Faculty & Research * 1. Overview 1. AI/Machine Learning 2. Metamaterials 3. Quantum Computing 4. Nanoelectronic Materials & Devices 5. Sensing & Imaging 6. Trustworthy Computing 2. Faculty Profiles 3. Awards & Recognition 4. Research News 5. Ask an Expert * About * 1. From the Chair 2. News 1. Media Coverage 2. Email Newsletter 3. Duke ECE Magazine 3. Events 1. Distinguished Speaker Series 2. Seminars 4. Facts & Stats 5. Mission & Vision 6. Diversity, Equity, Inclusion & Community 7. Entrepreneurship Success Stories 8. Meet Our Alumni 9. Industry Advisory Board",
                   "© Copyright 2011-2024 Duke University * __Main Menu * __Why Duke? * __On-Campus * __Duke MEM On-Campus * __Curriculum * __Curriculum Overview * __Seminar & Workshop Series * __Required Internship * __Consulting Practicum * __Elective Tracks * __Course Descriptions * __Flexible Degree Options * __Student Services and Support * __Career Outcomes * __Meet Our Alumni * __Tuition and Financial Aid * __External Funding Opportunities * __Online * __Duke MEM Online * __Boeing-Learning Together * __Curriculum * __Elective Tracks * __Residencies * __Course Descriptions * __Previous Courses * __Student Services and Support * __Meet Our Alumni * __Tuition and Financial Aid * __Certificate * __Apply * __Apply to Duke * __Application Requirements * __The 5 Principles * __Quick Links * __Directory * __Industry * __Alumni * __News * __Students * __Contact *",
                   "Jump to navigation * Duke Engineering * Pratt School of Engineering * Institute for Enterprise Engineering * Directory * Industry * Alumni * News * Students * Contact __ * Why Duke? * On-Campus * Duke MEM On-Campus * Curriculum * Elective Tracks * Course Descriptions * Flexible Degree Options * Student Services and Support * Career Outcomes * Meet Our Alumni * Tuition and Financial Aid * Online * Duke MEM Online * Boeing-Learning Together * Curriculum * Elective Tracks * Residencies * Course Descriptions * Student Services and Support * Meet Our Alumni * Tuition and Financial Aid * Certificate * Apply * Apply to Duke * Application Requirements * The 5 Principles __",
                   "Skip to main content * Duke University » * Pratt School of Engineering » ## Secondary Menu * Apply * Careers * Contact * Undergraduate * 1. Overview 2. Degree Programs 1. BSE Degree Planning 2. ME/BME Double Major 3. Certificates 4. 4+1: BSE+Master's 5. Courses 3. For Applicants 1. Why Duke MEMS? 2. Where Our Students Go 3. Enrollment and Graduation Rates 4. For Current Students 1. Awards & Honors 2. Graduation with Distinction 3. Independent Study 4. Senior Design * Master's * 1. Earn Your Master's at Duke 2. Admissions 3. Degrees 1. Master of Science 2. Master of Engineering 4. Concentrations 5. Certificates 1. Aerospace Graduate Certificate 6. Courses 7. Career Outcomes 8. Life at Duke 9. MEMS Graduate Student Committee * PhD * 1. Earn Your PhD at Duke 2. PhD Admissions 3. Certificates, Fellowships & Training Programs 4. Courses 5. Career Outcomes 6. Meet Our PhD Students 7. MEMS Graduate Student Committee * Research * 1. Overview 2. Aero 3. Autonomy 4. Bio 5. Computing / AI 6. Energy 7. Soft / Nano 8. Research Facilities * Faculty * 1. All Faculty 2. Awards & Recognition * About * 1. Welcome to Duke MEMS 2. Meet the Alstadt Chair 3. Meet the Staff 4. Facts & Stats 5. Diversity, Equity, Inclusion & Community 6. News 1. Media Coverage 2. Email Newsletter 3. Research News 7. All Events 1. Pearsall Lecture Series 2. Seminars 8. Our History 9. Driving Directions",
                   "Page Not Found We're sorry, but that page was not found. Please check the spelling of the page address or search this website. * * * * * © Copyright 2011-2024 Duke University drupal_block( 'search_form_block', { label_display: false } ) * Undergraduate * Overview * Degree Programs * BSE Degree Planning * ME/BME Double Major * Certificates * Aerospace Engineering * Energy & the Environment * Materials Science & Engineering * 4+1: BSE+Master's * Courses * For Applicants * Why Duke MEMS? * Where Our Students Go * Enrollment and Graduation Rates * For Current Students * Awards & Honors * Graduation with Distinction * Independent Study * Senior Design * Master's * Earn Your Master's at Duke * Admissions * Degrees * Master of Science * Master of Engineering * Concentrations * Certificates * Aerospace Graduate Certificate * Courses * Career Outcomes * Life at Duke * MEMS Graduate Student Committee * PhD * Earn Your PhD at Duke * PhD Admissions * Certificates, Fellowships & Training Programs * Courses * Career Outcomes * Meet Our PhD Students * MEMS Graduate Student Committee * Research * Overview * Aero * Autonomy * Bio * Computing / AI * Energy * Soft / Nano * Research Facilities * Faculty * All Faculty * Awards & Recognition * About * Welcome to Duke MEMS * Meet the Alstadt Chair * Meet the Staff * Facts & Stats * Diversity, Equity, Inclusion & Community * News * Media Coverage * Email Newsletter * Research News * All Events * Pearsall Lecture Series * Seminars * Our History * Driving Directions",
                   "© Copyright 2011-2024 Duke University drupal_block( 'search_form_block', { label_display: false } ) * Undergraduate * Overview * Degree Programs * BSE Degree Planning * ME/BME Double Major * Certificates * Aerospace Engineering * Energy & the Environment * Materials Science & Engineering * 4+1: BSE+Master's * Courses * For Applicants * Why Duke MEMS? * Where Our Students Go * Enrollment and Graduation Rates * For Current Students * Awards & Honors * Graduation with Distinction * Independent Study * Senior Design * Master's * Earn Your Master's at Duke * Admissions * Degrees * Master of Science * Master of Engineering * Concentrations * Certificates * Aerospace Graduate Certificate * Courses * Career Outcomes * Life at Duke * MEMS Graduate Student Committee * PhD * Earn Your PhD at Duke * PhD Admissions * Certificates, Fellowships & Training Programs * Courses * Career Outcomes * Meet Our PhD Students * MEMS Graduate Student Committee * Research * Overview * Aero * Autonomy * Bio * Computing / AI * Energy * Soft / Nano * Research Facilities * Faculty * All Faculty * Awards & Recognition * About * Welcome to Duke MEMS * Meet the Alstadt Chair * Meet the Staff * Facts & Stats * Diversity, Equity, Inclusion & Community * News * Media Coverage * Email Newsletter * Research News * All Events * Pearsall Lecture Series * Seminars * Our History * Driving Directions",
                   "Skip to main content * Duke University * Pratt School of Engineering * Apply Online * Visit * Contact __ * About * Is Duke Right for Me? * About the MEng Degree at Duke * Courses and Curriculum * Internship/Project * Career Services & Outcomes * Options for Current Duke Students * Non-Degree Candidates * Apply * How to Apply * Connect With Us * Visit Duke * Application Requirements * Application Deadlines * Apply Online * Tuition and Financial Aid __",
                   "© Copyright 2011-2024 Duke University * __Main Menu * __About * __Is Duke Right for Me? * __About the MEng Degree at Duke * __Courses and Curriculum * __Internship/Project * __Career Services & Outcomes * __Options for Current Duke Students * __Non-Degree Candidates * __Apply * __How to Apply * __Connect With Us * __Visit Duke * __Application Requirements * __Uploading a Transcript * __Grade Scale * __Short Answer Essays * __Resume * __Recommendations * __GRE Scores * __English Language Testing * __Application Fee * __Interview/Video Introduction * __Minimum Application Requirements * __International Applicants * __Deposit for Enrolling Students * __Submitting Final Transcripts * __Application Deadlines * __Apply Online * __Tuition and Financial Aid * __Quick Links * __Apply Online * __Visit * __Contact *",
                   "| Duke AI Master of Engineering",  "Jump to navigation",  "Enginerring Industry Relations Leadership News Contact Why Duke?",  "The Duke Difference Career Services Graduate Outcomes What Tech Leaders Are Saying Degree Certificate Courses Faculty Apply", "Engineering Industry Relations Leadership News Contact Why Duke?", "Context:"]

    
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=self.pinecone_api_key)
       
        if self.pinecone_index_name in self.pc.list_indexes().names():
            #print(self.pinecone_index_name)
            self.index = self.pc.Index(self.pinecone_index_name)
        else:
            self.create_pinecone()
    
    # Create the Pinecone store
    def create_pinecone(self):
        if self.verbose:
            print('creating pinecone index')
        if self.pinecone_index_name not in self.pc.list_indexes().names():
            print('of name ', self.pinecone_index_name)
            #self.index = self.pc.Index(self.pinecone_index_name)
            self.pc.create_index(
                name=self.pinecone_index_name,
                dimension= 1536,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-west-2"
                    )
                )
            if self.verbose:
                print('Created new pinecone index named', self.pinecone_index_name)
        self.index = self.pc.Index(self.pinecone_index_name)
        #self.load_and_process_json(json_file)

    # deletes a pinecone index
    def clear_pinecone(self, index_to_clear):
        self.pc.delete_index(index_to_clear)

    # adds processed and embedded chunks from a json file to a pinecone index 
    def populate_pinecone(self, json_file):
        self.load_and_process_json(json_file)


    def chunk_text(self, text):
        """
        Splits text into chunks with a specified maximum length and overlap,
        trying to split at sentence endings when possible.

        Imput:
            self
            text (str): The input text.
        Output:
            chunks (list of str): Chunks of text.
        """
        chunks = []
        current_chunk = ""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = current_chunk[-self.chunk_overlap:] + ' ' + sentence
            else:
                current_chunk += sentence + ' '

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def process_text(self, source, text, chunk_id):
        """
        Processes a grouping of the source, text, and chunk_id by getting embeddings
        and adding to the pinecone storage

        Imput:
            self
            source (str): The url of the source of the chunk with the hightest relevance
            text (str): The text of the chunk
            chunk_id (str): The number corresopnding to the chunk
        Output:
            No output. Adjusts the pinecone vector storage
        """
        
        unique_id = hashlib.sha256(f"{source}_{chunk_id}".encode()).hexdigest()
        response = self.openai_client.embeddings.create(
            model=self.openai_embedding_model,
            input=[text]
            )
        embedding = response.data[0].embedding
        self.index.upsert(vectors=[{"id": unique_id, "values":embedding, "metadata":{"source": source, "text": text}}])



    def load_and_process_json(self, json_file):
        """
        Loads the json file and calls the text chunker

        Imput:
            self
            json_file (str): The path/name of the json file to load and process
        Output:
            No output. Calls process text to add to the vector store
        """
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
        for source, text in data.items():
            print(f"Processing text: {source}")
            # cleaning text
            for phrase in self.text_to_replace:
                text = text.replace(phrase, " ")
                if self.verbose:
                    print('replacing phrase', phrase)
            chunks = self.chunk_text(text)
            if isinstance(chunks, list):
                for i, chunk in enumerate(chunks):
                    #print(i, chunk)
                    self.process_text(source, chunk, i)
            
            else:
                 print(chunks[0:40])
                 self.process_text(source, chunks, 0)


    def semantic_search(self, query):
        """
        Performs a semantic search on the pinecone vector database

        Imput:
            self
            query (str): The text input that is to be matched in the vector database
            
        Output:
            texts (list of str): the k highest matching chunks in a list
            sources (list of str): the source URLs of the k highest matching chunks
        """
        source_list = []
        texts = []
        try:
            #query_embedding = self.model.encode(query).tolist()
            response = self.openai_client.embeddings.create(
                model=self.openai_embedding_model,
                input=[query]
                )
            query_embedding = response.data[0].embedding


            results = self.index.query(vector=query_embedding, top_k=self.top_k, include_metadata=True)
            matches = [match for match in results["matches"]]
            for match in matches:
                source_list.append(match['metadata']['source'])
                texts.append(match['metadata']['text'])
            return texts, source_list
        except Exception as e:
            print(f"Error during semantic search: {e}")
            return [], []
        

    def generate_response(self, query):
        """
        Generates the response based on the provided query by calling the semantic search, 
        summarizing the top k chunks, and calling the function that sends the summarized chunks to
        either chatgpt or our fine-tuned model depending on self.use_gpt which can be toggled in the
        app. 
        Inputs:
            self
            query (str): The user input question as a string.
        Outputs:
            response, sources (str, str): The output string from the chosen model and associated source
        """
        texts, sources = self.semantic_search(query)
        if texts:
            combined_chunks = " ".join(texts)
            summarized_response = self.openai_client.chat.completions.create(
                    model=self.openai_engine,  
                    messages=[{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content":"Summarize the following text:\n" + combined_chunks + "Please provide a concise summary to prospective students who are seeking answers to questions, starting directly with the key points without introductory phrases like 'The text discusses', 'The text outlines', 'The text covers', 'The text provided' or 'The text introduces'. Please generate text in complete sentences. Please also do not over-summarized, student need useful information"}],
                    max_tokens=300,  
                    temperature=0.1  
                )
            summarized_chunks = summarized_response.choices[0].message.content
            prompt = self.prompt_instruction + "\n Context: " + summarized_chunks + "\nUser Query:\n\n {} ###\n\n".format(query)
            response = self.integrate_llm(prompt)
            return response, sources
        else:
            return ("Sorry, I couldn't find a relevant response.", None)

    def integrate_llm(self, prompt):
        """
        Handles getting the response from the chosen model and the interface with the model for the 
        prompt provided by generate_response

        Imput:
            self
            prompt (str): The text input that forms the prompt as constructed in generate_response         
        Output:
            chat_message (str): The output message from the model
        """
        if self.use_gpt:
            message=[{"role": "assistant", "content": "You are a trusted advisor helping to explain the text to prospective or current students who are seeking answers to questions"}, {"role": "user", "content": prompt}]
            if self.verbose:
                print('Debug: message is', message)
                print('using gpt is : ', self.use_gpt)
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.openai_engine,  
                    messages=message,
                    max_tokens=300,  
                    temperature=0.1  
                )
                # Extracting the content from the response
                chat_message = response.choices[0].message.content
                
                return chat_message
    
            except Exception as e:
                print(f"Error in generating response: {e}")
                return "Error in generating response"
        
        else:

            # Using the custom LLM HuggingFace endpoint
            headers = {
                "Authorization": f"Bearer {self.llm_token}"
            }
            payload = {
                "inputs": prompt
            }
            try:
                response = requests.post(self.llm_url, headers=headers, json=payload)
                response_data = response.json()
                response_text = response_data[0].get("generated_text", "No response generated.")
                response_text = response_text.replace(self.prompt_instruction, ' ').replace('#', ' ').split('User Query')[0]
                for phrase in self.text_to_replace:
                    response_text = response_text.replace(phrase, ' ')

                
                return response_text
            except Exception as e:
                print(f"Error in connecting to the HuggingFace API: {e}")
                return "Error in connecting to the HuggingFace API. Please wait a few minutes or try using GPT (toggle above). Additionally, you can click the 'View Source' button to view a relevant web page."


    def get_similar_faq(self, query, index_name="faq-database", threshold=0.60):
        '''
        Gets a QA pair from the FAQ database with the highest similarity score if above the threshold value.
        Inputs:
            query(str): the entered user question
            index_name(str): the pinecone index
            threshold(float): the threshold value above which is considered a match on the similarity search
        Outputs:
            matching_text(str): the answer for the matching query
            matching_score(float): the similarity score for the matching item
        '''
        #print('get similar faq running with text', query)
        try:
            # Initialize Pinecone client
            faq_pc = Pinecone(self.pinecone_api_key)
            if self.verbose:
                print('setting up pinecone client')
            # Get the index
            faq_index = faq_pc.Index(index_name)
            if self.verbose:

                print('Setting up pinecone client with index', index_name)

            #Embed the query using OpenAI's embedding API
            response = self.openai_client.embeddings.create(
                model=self.openai_embedding_model,
                input=[query])
            #print('response is', response)
            query_embedding = response.data[0].embedding
            #print('query_embedding finished.', query_embedding)
            
            # Query the vector index
            results = faq_index.query(vector=query_embedding, top_k=1, include_metadata=True)
            matches = results["matches"]
            #print('matches:', matches)
            if matches and matches[0]["score"] >= threshold:
                # returns the text and the score
                matching_text = matches[0]["metadata"]["answer"]
                matching_score = matches[0]["score"]
                return matching_text, matching_score
            else:
                return None, None

        except Exception as e:
            print(f"Error: {e}")
            return None, None

# Example usage with command-line argument for specifying the JSON file
if __name__ == "__main__":
    rag = RAG()
    parser = argparse.ArgumentParser(description="Load text data from a JSON file and process with RAG.")
    parser.add_argument("--json_file", default="data/extracted_data_2024-04-01_07-59-36.json", help="Path to the JSON file containing text data.")
    args = parser.parse_args()

    # Initialize your RAG instance 
    rag = RAG()
    #rag.populate_pinecone('data/extracted_data_2024-04-01_07-59-36.json')

    # Load and process the specified JSON file for creating the vector db. 
    # Not necessary if already created
    

    # Query the pinecone vector storage
    phrase = 'How do I get a roommate?'
    #print('phrase is', phrase)
    #texts, sources = rag.semantic_search(phrase)
    #print(texts)
    similar_faq, similar_score = rag.get_similar_faq(phrase)
    #print(similar_faq)
    print('Matching text: ', similar_faq, '\nScore: ', similar_score)
    

#    llm_response, sources = rag.generate_response(phrase)
#    print(llm_response)
#    print('learn more by clicking', sources[0])
