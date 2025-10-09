import os, base64
from openai import OpenAI
from text_extraction import extract_text_from_file

from dotenv import load_dotenv
import os

load_dotenv()  # loads from .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)


def process_file_with_llm(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    try:
        # -----------------------------
        # Text-based files: PDF / DOCX / TXT
        # -----------------------------
        if ext in (".pdf", ".docx", ".txt"):
            # extract_text_from_file must exist in your notebook
            raw = extract_text_from_file(file_path)
            if not raw or raw.strip() == "":
                raw = "No readable content found."

            # 1) factual summary (low randomness)
            resp_sum = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a precise legal summarizer for subrogation demand packages."},
                    {"role": "user", "content": (
                        "Produce a concise factual summary for inclusion in a demand package. "
                        "Keep to 3 short bullets (facts only - who/what/when/damages/liability). "
                        "Do NOT include recommendations or internal notes.\n\n"
                        f"EXHIBIT CONTENT:\n{raw[:4500]}"
                    )}
                ],
                temperature=0.1,
                top_p=0.1,
                max_tokens=600
            )
            summary = resp_sum.choices[0].message.content.strip()

            # 2) top 2-3 follow-up questions (internal)
            resp_fup = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are assisting a subrogation adjuster."},
                    {"role": "user", "content": (
                        "Provide the top 2–3 follow-up questions an adjuster should ask based on the exhibit. "
                        "Return ONLY the questions as a short numbered list (no explanation).\n\n"
                        f"EXHIBIT CONTENT:\n{raw[:3500]}"
                    )}
                ],
                temperature=0.2,
                top_p=0.5,
                max_tokens=300
            )
            followups = resp_fup.choices[0].message.content.strip()

            # 3) top 2-3 recommended next steps (internal)
            resp_recs = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a subrogation strategist giving actionable next steps."},
                    {"role": "user", "content": (
                        "Provide the top 2–3 recommended next steps for the adjuster (short, numbered list). "
                        "Be actionable (e.g., verify X, obtain Y, contact Z).\n\n"
                        f"EXHIBIT CONTENT:\n{raw[:3500]}"
                    )}
                ],
                temperature=0.35,
                top_p=0.9,
                max_tokens=300
            )
            recommendations = resp_recs.choices[0].message.content.strip()

        # -----------------------------
        # Image files: use vision model for summary
        # -----------------------------
        elif ext in (".png", ".jpg", ".jpeg"):
            with open(file_path, "rb") as f:
                b64_img = base64.b64encode(f.read()).decode("utf-8")

            # 1) image factual description with a stronger vision model
            resp_vis = client.chat.completions.create(
                model="gpt-4.1",   # image-capable model (per your request)
                messages=[
                    {"role": "system", "content": "You are a legal assistant specialized in photo evidence."},
                    {"role": "user", "content": [
                        {"type": "text", "text": (
                            "Describe this photo evidence in factual terms suitable for an external demand package. "
                            "Focus on what is visible (vehicle types, damage location, environmental conditions). "
                            "Keep to 3 concise factual bullets."
                        )},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                    ]}
                ],
                temperature=0.1,
                top_p=0.1,
                max_tokens=700
            )
            summary = resp_vis.choices[0].message.content.strip()

            # 2) follow-ups based on image (2 top questions)
            resp_fup = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are assisting a subrogation adjuster."},
                    {"role": "user", "content": (
                        "Based on the photo description above, list the top 2 follow-up questions an adjuster should ask. "
                        "Return only the numbered questions."
                    )}
                ],
                temperature=0.2,
                top_p=0.5,
                max_tokens=200
            )
            followups = resp_fup.choices[0].message.content.strip()

            # 3) recommendations based on image (2 top steps)
            resp_recs = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a litigation strategist."},
                    {"role": "user", "content": (
                        "Based on the photo description above, provide the top 2 recommended next steps for the adjuster. "
                        "Return as a short numbered list."
                    )}
                ],
                temperature=0.35,
                top_p=0.9,
                max_tokens=200
            )
            recommendations = resp_recs.choices[0].message.content.strip()

        else:
            summary = "Unsupported file type for processing."
            followups = "N/A"
            recommendations = "N/A"

        # Ensure we always return three values
        return summary, followups, recommendations

    except Exception as err:
        # Return safe fallback strings (so create_final_reports won't crash)
        return f"[LLM error: {err}]", "N/A", "N/A"





def llm(claims_notes):

    topic_que = {
        'Injury at scene' : ['Is the injury reported on scene?'
                            'Is there any medical treatment received?',
                            'Is there an ambulance on scene?',
                            'What is the time duration between loss reported and receiving medical treatment?'],

        'Photographs' :['Is there any photograph of the vehicle?'
                        'What is th extent of damage to the yehicle?',
                        'Are the photographs admissable?'],

        'Mechanism of injury' : ['Is there any medical testimony available?',
                                'Are the plaintiffs injuries disputed by credible medical or biomechanical testimony?'],
        'Venue':['ls the venue plaintiff friendly?'],
        'Medical treatment issues' :['Is there a siginificant gap in medical treatments?'],
        'Pre-existing conditions':['Is the plaintiff an egg-shell plaintiff?',
                                    'Are the plaintiffs pre-existing conditions related to the accident?'],

        'Plaintiffs Attorney/ Law Firm' :['Is the plaintiffs attorney an exceptional litigator?',
                                            'Does Plaintiffs counsel handles a high volume of lawsuits?'],

        'Police report' :['Is the police report available?'],
        'Body part injury':['Has the plaintiff received any injury?','What is the extent of injury received?'],
        'Driver':['Is the driver insured?','Is the driver available for testimony?'],
        'Vehicle':['Is the vehicle insured or rental?','What is the number of vehicles involved in the accident?'],
        'State':['Is the loss state different from the jurisdiction state?'],
        'ISO':['Does the plaintiff have any meaningful ISO entries?','Does the medical provider have any meaningful ISO entries?'],
        'Statute of Limitations': ['Has the lawsuit been filed within the applicable statute of limitations?']


    }

    # Formatting the topics and questions into a clear string for the prompt
    topics_and_questions_str = ""
    for topic, questions in topic_que.items():
        topics_and_questions_str += f"\n### {topic}\n"
        for q in questions:
            topics_and_questions_str += f"- {q}\n"
    

    # --- Restructuring the prompt. Task description is at the starting of the prompt ---
    prompt_instructions = f"""### CORE TASK
    Your job is to thoroughly analyze the following CLAIM NOTES to identify and list all potential litigation indicators. Structure your analysis based on the TOPICS & QUESTIONS provided.

    ### CLAIM NOTES TO ANALYZE
    {claims_notes}

    ### TOPICS & QUESTIONS TO ADDRESS
    {topics_and_questions_str}

    ### OUTPUT FORMATTING RULES
    - For each topic, compile a single, concise list of all identified potential litigation indicators.
    - If a relevant red flag for a specific question is found, include that specific red flag in the list.
    - **Do not include the original question in the output.** Instead, directly state the identified indicator.
    - If no information or red flag is found for any question within a topic, explicitly state "No indicators found in claim notes." for that entire topic.
    - The output for each topic should begin with the topic name itself on its own line (e.g., Jurisdiction). Do not add the prefix 'Topic:'.
    - Use only bullet points (•). Do not use numbers or paragraphs.
    - Do not use any generic information or information outside the provided claim notes.
    - Do not provide suggestions, recommendations, conclusions, or opinions. Only state facts from the notes.
    - Do not change or manipulate any fact.
    """



    try:
    
        # 1) factual summary (low randomness)
        resp_sum = client.chat.completions.create(
            model="gpt-4-0613",
            messages=[
                {"role": "system", "content": "You are an insurance claims assistant"},
                {"role": "user", "content": prompt_instructions }
                
            ],
            temperature=0.1,
            top_p=0.1,
            #max_tokens=600
        )
        summary = resp_sum.choices[0].message.content.strip()
        
        return summary

    except Exception as err:
        # Return safe fallback strings (so create_final_reports won't crash)
        return f"[LLM error: {err}]", "N/A", "N/A"