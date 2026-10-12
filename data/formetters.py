import datasets

_RAG_SYSTEM_PROMPT = """You are a helpful assistant designed to generate personalized responses to user questions.
Your task is to answer a user's question from a post in a personalized way by considering this user's past post questions and detailed descriptions of these questions.

# Your Input:

- The user's current question from a post.

- The user's past post questions and detailed descriptions of these questions.

# Your Task:
Answer the user's current question in a personalized way by considering the user's past post questions and detailed descriptions of these questions, to learn about the user's preferences.

# Your Output:
Your output must be a valid JSON object in a ```json ``` block containing the following field:

- personalized_answer: a paragraph of text that contains the personalized answer to the user's current question considering the user's preferences.

```json
{
  "personalized_answer": "..."
}
```
"""

_RAG_WITH_PLAN_SYSTEM_PROMPT = """You are a helpful assistant designed to generate personalized responses to user questions. Your task is to answer a user's question from a post in a personalized way by considering this user's past post questions and detailed descriptions of these questions. Additionally, you are provided with the aspects that the user expects to see in the response to their question, which you can use to generate a personalized answer.

# Your input:
    - The user's current question from a post.
    - The user's past post questions and detailed descriptions of these questions.
    - The aspects that the user expects to see in the response to their question.
# Your task: Answer the user's current question in a personalized way by considering this user's past post questions and detailed descriptions of these questions, to learn about the user's preferences. Additionally, you should consider the aspects that the user expects to see in the response to their question.
# Your output: You should generate personalized answer to the user's current question by considering this user's past post questions and detailed descriptions of these questions to learn about user's preferences. Additionally, you should consider the aspects that the user expects to see in the response to their question. Your output should be a valid json object in ```json ``` block that contains the following fields: 
    - personalized_answer: contains the personalized answer to the user's current question considering the this user's past post questions and detailed descriptions of these questions to learn about user's preferences.
"""

_PLANNER_PROMPT = """You are a helpful assistant designed to generate the topics that user might expect to see in a response to their question. Your task is to extract the important aspects that the user expects to see in a response to their question considering the previous questions asked by the same user and the detailed information need they provided in the post.
# Your input:
    - The user's current question.
    - The user's past post questions and detailed descriptions of these questions.
# Your task: Extract the important aspects that the user expects to see in a response to their question considering the previous questions asked by the same user and the detailed information need they provided in the post.
# Your output: You should generate a list of aspects that are important for the user to be included in the response to their question. 
"""
_RAG_USER_PROMPT = """
# Past post questions and detailed descriptions of these questions:
{profile}
# Current post question:
{question}
"""

_RAG_WITH_PLAN_USER_PROMPT = """
# Past post questions and detailed descriptions of these questions:
{profile}
# Current post question:
{question}
# Aspects expected in the response:
{aspects}
"""


def apply_chat_template(conversation, tokenizer, tokenize=True, add_generation_prompt=False):
    try:
        return tokenizer.apply_chat_template(conversation, tokenize=tokenize,
                                             add_generation_prompt=add_generation_prompt)
    except Exception as e:
        if e.message == "System role not supported":
            conversation_new = []
            system_prompt = conversation[0]['content']
            user_prompt = conversation[1]['content']
            new_user_prompt = system_prompt + "\n\n" + user_prompt
            conversation_new.append({"role": "user", "content": new_user_prompt, "type": "text"})
            return tokenizer.apply_chat_template(conversation_new, tokenize=tokenize,
                                                 add_generation_prompt=add_generation_prompt)
        else:
            raise e

def prepare_prompts(dataset, formater):
    reshaped_dataset = {
        "question": [],
        "id": [],
        "profile": []
    }
    for data in dataset:
        reshaped_dataset["question"].append(data["question"])
        reshaped_dataset["id"].append(data["id"])
        reshaped_dataset["profile"].append(data["profile"])
    return formater(reshaped_dataset)

def get_baseline_no_rag_formatter(tokenizer, proprietary_llm=False):
    def formatter(data):
        texts = []
        for i in range(len(data['question'])):
            user_prompt = data['question'][i]
            conversation = [
                {
                    "role": "system",
                    "content": """You are a helpful assistant designed to generate personalized responses to user questions.
Your output must be a valid JSON object in a ```json ``` block containing the following field:
- personalized_answer: a paragraph of text that contains the personalized answer to the user's current question considering the user's preferences.""",
                    "type": "text"
                },
                {
                    "role": "user",
                    "content": user_prompt,
                    "type": "text"
                }
            ]
            if proprietary_llm:
                text = conversation
            else:
                text = apply_chat_template(conversation, tokenizer, tokenize=False, add_generation_prompt=True)
            texts.append(text)
        return texts

    return formatter


def get_baseline_rag_formatter(tokenizer, num_contexts, proprietary_llm=False, train=False):
    def formatter(data):
        texts = []
        for i in range(len(data['question'])):
            user_prompt = data['question'][i]
            profile = "\n\n".join([x['text'] for x in data['profile'][i][:num_contexts]])
            conversation = [
                {
                    "role": "system",
                    "content": _RAG_SYSTEM_PROMPT,
                    "type": "text"
                },
                {
                    "role": "user",
                    "content": _RAG_USER_PROMPT.format(profile=profile, question=user_prompt),
                    "type": "text"
                }
            ]
            if proprietary_llm:
                text = conversation
            else:
                if train:
                    expeted_output = data['expected_output'][i]
                    conversation.append({
                        "role": "assistant",
                        "content": expeted_output,
                        "type": "text"
                    })
                    text = {"messages": conversation}
                else:
                    text = apply_chat_template(conversation, tokenizer, tokenize=False, add_generation_prompt=True)
            texts.append(text)
        if train:
            dataset = datasets.Dataset.from_list(texts)
            return dataset
        return texts

    return formatter


def get_planner_formatter(tokenizer, num_contexts, train=False, proprietary_llm=False):
    def formatter(data):
        texts = []
        for i in range(len(data['question'])):
            user_prompt = data['question'][i]
            profile = "\n\n".join([x['text'] for x in data['profile'][i][:num_contexts]])
            conversation = [
                {
                    "role": "system",
                    "content": _PLANNER_PROMPT,
                    "type": "text"
                },
                {
                    "role": "user",
                    "content": _RAG_USER_PROMPT.format(profile=profile, question=user_prompt),
                    "type": "text"
                }
            ]
            if not train:
                if proprietary_llm:
                    text = conversation
                else:
                    text = apply_chat_template(conversation, tokenizer, tokenize=False, add_generation_prompt=False)
                texts.append(text)
            else:
                aspects = ""
                for aspect in data['rubric_aspects'][i]:
                    aspects += "- " + aspect['aspect'] + "\n"
                conversation.append({
                    "role": "assistant",
                    "content": f"The user expects to see the following aspects in the response to their question:\n{aspects}",
                    "type": "text"
                })
                texts.append({"messages": conversation})
        if train:
            dataset = datasets.Dataset.from_list(texts)
            return dataset
        return texts

    return formatter


def get_generation_with_plan_rag_formatter(tokenizer, num_contexts, proprietary_llm=False):
    def formatter(data):
        texts = []
        for i in range(len(data['question'])):
            user_prompt = data['question'][i]
            profile = "\n\n".join([x['text'] for x in data['profile'][i][:num_contexts]])
            plan = data['plan'][i]
            conversation = [
                {
                    "role": "system",
                    "content": _RAG_WITH_PLAN_SYSTEM_PROMPT,
                    "type": "text"
                },
                {
                    "role": "user",
                    "content": _RAG_WITH_PLAN_USER_PROMPT.format(profile=profile, question=user_prompt, aspects=plan),
                    "type": "text"
                }
            ]
            if proprietary_llm:
                text = conversation
            else:
                text = apply_chat_template(conversation, tokenizer, tokenize=False, add_generation_prompt=True)
            texts.append(text)
        return texts

    return formatter

_ALL_RAG_SYSTEM_PROMPT = """You are a helpful assistant designed to generate personalized responses to user questions. Your task is to answer a user's question from a post in a personalized way by considering this user's past post questions and detailed descriptions of these questions.
# Your input:
    - The user's current question from a post.
    - The user's past post questions and detailed descriptions of these questions.
    - The response of the larger model to the same question.
# Your task: Answer the user's current question in a personalized way by considering this user's past post questions and detailed descriptions of these questions, to learn about the user's preferences.
# Your output: You should generate personalized answer to the user's current question by considering this user's past post questions and detailed descriptions of these questions to learn about user's preferences. Your output should be a valid json object in ```json ``` block that contains the following fields:
    - personalized_answer: contains the personalized answer to the user's current question considering the this user's past post questions and detailed descriptions of these questions to learn about user's preferences.
"""
_ALL_RAG_USER_PROMPT = """
# Past post questions and detailed descriptions of these questions:
{profile}
# The larger model's response to this question:
{llm_output}
# Current post question:
{question}
"""
_ALL_USER_PROMPT = """
# The LLM's response to this question:
{llm_output}
# Current post question:
{question}
"""
def get_all_rag_formatter(tokenizer, num_contexts, proprietary_llm=False, train=False):
    def formatter(data):
        texts = []
        for i in range(len(data['question'])):
            user_prompt = data['question'][i]
            profile = "\n\n".join([x['text'] for x in data['profile'][i][:num_contexts]])
            conversation = [
                {
                    "role": "system",
                    "content": _ALL_RAG_SYSTEM_PROMPT,
                    "type": "text"
                },
                {
                    "role": "user",
                    "content": _ALL_RAG_USER_PROMPT.format(profile=profile, llm_output=data['llm_output'][i], question=user_prompt),
                    "type": "text"
                }
            ]
            if proprietary_llm:
                text = conversation
            else:
                if train:
                    expeted_output = data['expected_output'][i]
                    conversation.append({
                        "role": "assistant",
                        "content": expeted_output,
                        "type": "text"
                    })
                    text = {"messages": conversation}
                else:
                    text = apply_chat_template(conversation, tokenizer, tokenize=False, add_generation_prompt=True)
            texts.append(text)
        if train:
            dataset = datasets.Dataset.from_list(texts)
            return dataset
        return texts

    return formatter

def get_all_no_rag_formatter(tokenizer, proprietary_llm=False):
    def formatter(data):
        texts = []
        for i in range(len(data['question'])):
            user_prompt = data['question'][i]
            conversation = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant designed to generate personalized responses to user questions.  Your output should be a valid json object in ```json ``` block that contains the following fields:\n   - personalized_answer: contains the personalized answer to the user's current question.",
                    "type": "text"
                },
                {
                    "role": "user",
                    "content": _ALL_USER_PROMPT.format(llm_output=data['llm_output'][i], question=user_prompt),
                    "type": "text"
                }
            ]
            if proprietary_llm:
                text = conversation
            else:
                text = apply_chat_template(conversation, tokenizer, tokenize=False, add_generation_prompt=True)
            texts.append(text)
        return texts

    return formatter

def prepare_all_prompts(dataset, dataset_llm, formater):
    reshaped_dataset = {
        "question": [],
        "id": [],
        "profile": [],
        "llm_output": [],
    }
    for data, data_llm_idx in zip(dataset, dataset_llm.keys()):
        reshaped_dataset["question"].append(data["question"])
        reshaped_dataset["id"].append(data["id"])
        reshaped_dataset["profile"].append(data["profile"])
        reshaped_dataset["llm_output"].append(dataset_llm[data_llm_idx][0]["output"])
    return formater(reshaped_dataset)

_STEP0_SYSTEM_PROMPT = """You are a helpful assistant designed to analyze and summarize historical information from user's past posts to enable personalized responses to current questions. Your task is to analyze the user's past post questions and detailed descriptions, and generate a summary of the relevant historical information that can be used to answer the user's current question in a personalized way.
# Your input:
    - The user's current question from a post.
    - The user's past post questions and detailed descriptions of these questions.
# Your task: Analyze the user's past post questions and detailed descriptions to identify and summarize the historical information that is relevant to the user's current question, in order to learn about the user's preferences and enable a personalized response. 
# Your output: You should generate a summary of the relevant historical information that can be used to personalize the answer to the user's current question. The summary must be strictly under 150 words and must not disclose any private information. Your output should be a valid JSON object in a ```json ``` block that contains the following field:
    - historical_summary: contains the summary text of the relevant historical information, based on the user's past post questions and descriptions.
"""
_STEP0_USER_PROMPT = """
# Past post questions and detailed descriptions of these questions:
{profile}
# Current post question:
{question}
"""
def get_step0_context_formatter(tokenizer, num_contexts, proprietary_llm=False, train=False):
    def formatter(data):
        texts = []
        for i in range(len(data['question'])):
            user_prompt = data['question'][i]
            profile = "\n\n".join([x['text'] for x in data['profile'][i][:num_contexts]])
            conversation = [
                {
                    "role": "system",
                    "content": _STEP0_SYSTEM_PROMPT,
                    "type": "text"
                },
                {
                    "role": "user",
                    "content": _STEP0_USER_PROMPT.format(profile=profile, question=user_prompt),
                    "type": "text"
                }
            ]
            if proprietary_llm:
                text = conversation
            else:
                if train:
                    expeted_output = data['expected_output'][i]
                    conversation.append({
                        "role": "assistant",
                        "content": expeted_output,
                        "type": "text"
                    })
                    text = {"messages": conversation}
                else:
                    text = apply_chat_template(conversation, tokenizer, tokenize=False, add_generation_prompt=True)
            texts.append(text)
        if train:
            dataset = datasets.Dataset.from_list(texts)
            return dataset
        return texts

    return formatter

_CLARIFY_SYSTEM_PROMPT = """You are a helpful assistant designed to clarify the user's current question using private personalized information when the question is ambiguous, contains vague references or pronouns, or needs missing details that appear in the private context.

# Your Input:

- The user's current question

- The description of user's private personalized information

# Your Task:
Identify ambiguous or vague elements; use the personalized information to resolve them. Generate a single self-contained clarified question that preserves the user's intent.

# Your Output:
Your output must be a valid JSON object in a ```json ``` block containing the following field:

- refined_question: the clarified question.

```json
{
  "refined_question": "..."
}
```
"""

_DECOMPOSE_SYSTEM_PROMPT = """You are a helpful assistant designed to decompose user questions into sub-questions. Your task is to break down the user's current question into two independent sub-questions based on their personalized context, and select one for the cloud LLM.
# Your input:
- The user's current question
- The description of user's private personalized information
# Your task:
Analyze the user's current question and personalized information, decompose the question into exactly TWO sub-questions that together address the original question. 
# Your output: Your output must be a valid JSON object in a ```json``` block that contains the following field:
    - refined_question: 
        - STEP 1: [Harder sub-question, cloud-executable, no local data] 
        - STEP 2: [Simpler sub-question, edge-executable, needs local data] 
"""

_KNOWLEDGE_SYSTEM_PROMPT = """You are a helpful assistant designed to identify gaps in general knowledge needed to answer the user's current question.
Your task is to generate a question that requests general knowledge from a more knowledgeable LLM, based on the user's current question and personalized information.

# Your Input:

- The user's current question

- The description of user's private personalized information

# Your Task:
Analyze the user's current question and personalized information to identify what general knowledge is missing to fully address the query.
Generate a single, clear question that requests the needed general knowledge.

# Your Output:
A single general knowledge question that requests broad, factual information relevant to the user's context.
Your output must be a valid JSON object in a ```json ``` block containing the following field:

- refined_question: a single general knowledge question that requests broad, factual information relevant to the user's context.

```json
{
  "refined_question": "..."
}
```
"""

_VERIFY_SYSTEM_PROMPT = """You are a helpful assistant designed to generate a simple process description for completing the task based on the user's question, while strictly protecting privacy.

# Your Input:

- The user's current question

- The description of user's private personalized information

# Your Task:
Generate a response that describes the general process to address the question using the private information, without revealing any private details.

# Your Output:
Your output must be a valid JSON object in a ```json ``` block containing the following field:

- answer: a simple, privacy-preserving process description under 150 words.

```json
{
  "answer": "..."
}
```
"""

_QUESTION_USER_PROMPT = """
# The description of user's private personalized information:
{profile}
# Current post question:
{question}
"""
refine_to_prompt = {
    "clarify": _CLARIFY_SYSTEM_PROMPT,
    "knowledge": _KNOWLEDGE_SYSTEM_PROMPT,
    "verify": _VERIFY_SYSTEM_PROMPT,
}

def prepare_refine_prompts(dataset, dataset_context, formater):
    reshaped_dataset = {
        "question": [],
        "id": [],
        "profile": [],
    }
    for data, data_cxt_idx in zip(dataset, dataset_context.keys()):
        assert data["id"] == data_cxt_idx
        reshaped_dataset["question"].append(data["question"])
        reshaped_dataset["id"].append(data["id"])
        reshaped_dataset["profile"].append(dataset_context[data_cxt_idx][0]["output"])
    return formater(reshaped_dataset)

def get_refine_formatter(tokenizer, refine_type, proprietary_llm=False, train=False):
    def formatter(data):
        texts = []
        for i in range(len(data['question'])):
            user_prompt = data['question'][i]
            profile = data['profile'][i]
            conversation = [
                {
                    "role": "system",
                    "content": refine_to_prompt[refine_type],
                    "type": "text"
                },
                {
                    "role": "user",
                    "content": _QUESTION_USER_PROMPT.format(profile=profile, question=user_prompt),
                    "type": "text"
                }
            ]
            if proprietary_llm:
                text = conversation
            else:
                if train:
                    expeted_output = data['expected_output'][i]
                    conversation.append({
                        "role": "assistant",
                        "content": expeted_output,
                        "type": "text"
                    })
                    text = {"messages": conversation}
                else:
                    text = apply_chat_template(conversation, tokenizer, tokenize=False, add_generation_prompt=True)
            texts.append(text)
        if train:
            dataset = datasets.Dataset.from_list(texts)
            return dataset
        return texts

    return formatter

SCORE_SYSTEM_PROMPT = """You are an intelligent intent classifier. Your task is to analyze the user's input and private context, then select the single best strategy to handle the request.

# Strategies Definition:
1. PLAN: Select this if the user's question contains pronouns (it, that, he), vague references, or is ambiguous, and the private context contains the missing details needed to make the question specific.
2. FACT: Select this if the user asks for general facts, definitions, history, or instructions that are NOT specific to their private data.
3. REFLECT: Select this if the user asks for a specific action, process, or piece of data that exists within the provided private context. The goal is to answer or act based on the private info.
4. ORIGIN: Select this if the user's question is very easy to handle and fully relies on private data.

Step 1 — CLARIFY (Answerability Check)
First determine whether the question is answerable as written.
Select CLARIFY if:
- The question contains ambiguous pronouns (it, that, he, she, they, this, etc.)
- The question is vague or underspecified
- The private context contains missing details required to make the question specific

If Step 1 applies, output PLAN.

Step 2 — FACT (Knowledge Gap Check)
If the question is clear and answerable, ask yourself:
Does answering this require general world knowledge, factual information, definitions, explanations, or domain knowledge not contained in the private context?

If YES → output FACT.

Step 3 — VERIFY (Private Context Dependency Check)
If no external knowledge is required, ask:
Does the answer require checking, retrieving, or operating on specific data inside the private context?

If YES → output REFLECT.

Step 4 — ORIGIN (Direct & Trivial Use of Context)
If none of the above apply:
- The question is simple
- It directly relies on private context
- No clarification, external knowledge, or verification reasoning is needed

Then output ORIGIN.

# Your Input:
- The description of user's private personalized information
- Current post question

# Your Output:
Based on the user's input, predict the suitability of each strategy.
You must end your response by outputting exactly one single word indicating the best strategy.
Choose from: PLAN, FACT, REFLECT, ORIGIN
Do not output any additional explanation. Just the word.
""".strip()

QUESTION_USER_PROMPT = """
# The description of user's private personalized information
{profile}
# Current post question
{question}
"""

_SCORE_USER_PROMPT = QUESTION_USER_PROMPT

def get_choose_formatter(tokenizer, proprietary_llm=False, train=False, num_contexts=10):
    def formatter(data):
        texts = []
        for i in range(len(data['question'])):
            user_prompt = data['question'][i]
            profile = "\n\n".join([x['text'] for x in data['profile'][i][:num_contexts]])
            conversation = [
                {
                    "role": "system",
                    "content": SCORE_SYSTEM_PROMPT,
                    "type": "text"
                },
                {
                    "role": "user",
                    "content": _SCORE_USER_PROMPT.format(profile=profile, question=user_prompt),
                    "type": "text"
                }
            ]
            if proprietary_llm:
                text = conversation
            else:
                if train:
                    expeted_output = data['expected_output'][i]
                    conversation.append({
                        "role": "assistant",
                        "content": expeted_output,
                        "type": "text"
                    })
                    text = {"messages": conversation}
                else:
                    text = apply_chat_template(conversation, tokenizer, tokenize=False, add_generation_prompt=True)
            texts.append(text)
        if train:
            dataset = datasets.Dataset.from_list(texts)
            return dataset
        return texts

    return formatter

def prepare_prompts_dict(dataset, dataset_profile, formater):
    reshaped_dataset = {
        "question": [],
        "id": [],
        "profile": []
    }
    for data in dataset_profile:
        data_idx = data["id"]
        if data_idx not in dataset.keys() or "question" not in dataset[data_idx][0].keys():
            continue
        reshaped_dataset["question"].append(dataset[data_idx][0]["question"])
        reshaped_dataset["id"].append(data_idx)
        reshaped_dataset["profile"].append(data["profile"])
    return formater(reshaped_dataset), reshaped_dataset["id"]


OVERALL_EVALUATOR_PROMPT = """Act as an impartial evaluator and evaluate how well the answer matches the user's demand and its relevance to the question.
Evaluate factors such as the response's consistency with their profile, helpfulness, relevance, accuracy, depth, creativity, and level of detail.

First, provide a concise analysis, strictly under 50 words.
You must then rate the response on a scale of 1 to 10 by strictly following this format: "Rating: [[rating]]", for example: "Rating: [[5]]".

# Your Input:

- [User Writing History] {writing_history}

- [Question] {question}

- [Answer] {answer}

# Your Task:
Evaluate the answer's alignment with the user's demand and relevance to the question.
Then output the rating in the required format.

# Your Output:
A concise analysis under 50 words, followed by one line in the format Rating: [[X]] where X is an integer from 1 to 10.
"""

PERSONALIZATION_EVALUATOR_PROMPT = """Act as an impartial evaluator and evaluate the answer response based on its alignment with the user's writing history.
Focus your assessment on the personalization aspects of the response, considering their preferences, context, and needs.

First, provide a concise analysis, strictly under 50 words.
You must then rate the response on a scale of 1 to 10 by strictly following this format: "Rating: [[rating]]", for example: "Rating: [[5]]".

# Your Input:

- [User Writing History] {writing_history}

- [Question] {question}

- [Answer] {answer}

# Your Task:
Evaluate how well the answer aligns with the user's writing history and personalization needs.
Then output the rating in the required format.

# Your Output:
A concise analysis under 50 words, followed by one line in the format Rating: [[X]] where X is an integer from 1 to 10.
"""