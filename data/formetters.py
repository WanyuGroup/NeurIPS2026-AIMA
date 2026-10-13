"""AIMA prompts and inference formatting only."""

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


_RAG_USER_PROMPT = """
# Past post questions and detailed descriptions of these questions:
{profile}
# Current post question:
{question}
"""


def apply_chat_template(conversation, tokenizer, tokenize=True, add_generation_prompt=False):
    try:
        return tokenizer.apply_chat_template(conversation, tokenize=tokenize,
                                             add_generation_prompt=add_generation_prompt)
    except Exception as e:
        if "System role not supported" in str(e):
            conversation_new = []
            system_prompt = conversation[0]['content']
            user_prompt = conversation[1]['content']
            new_user_prompt = system_prompt + "\n\n" + user_prompt
            conversation_new.append({"role": "user", "content": new_user_prompt, "type": "text"})
            return tokenizer.apply_chat_template(conversation_new, tokenize=tokenize,
                                                 add_generation_prompt=add_generation_prompt)
        else:
            raise e


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


def get_answer_formatter(tokenizer, num_contexts, proprietary_llm=False):
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
                text = apply_chat_template(conversation, tokenizer, tokenize=False, add_generation_prompt=True)
            texts.append(text)
        return texts

    return formatter
