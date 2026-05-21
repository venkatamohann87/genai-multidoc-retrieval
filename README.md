## Design and Implementation of a Multidocument Retrieval Agent Using LlamaIndex

### AIM:
To design and implement a multidocument retrieval agent using LlamaIndex to extract and synthesize information from multiple research articles, and to evaluate its performance by testing it with diverse queries, analyzing its ability to deliver concise, relevant, and accurate responses.

### PROBLEM STATEMENT:

Design an AI agent using LlamaIndex and OpenAI GPT-3.5 Turbo to answer questions from multiple research papers. The system should download papers, create document tools, build an indexed retriever, and use a function-calling agent to query and compare information from the papers Self-RAG, SWE-Bench, and LongLoRA.

### DESIGN STEPS:

#### STEP 1:
Download the research papers (PDF files) from the given URLs.
#### STEP 2:
Create tools for each paper to enable searching and summarizing the content.
#### STEP 3:

Build an AI agent, connect it with the tools, and ask questions to get summaries and comparisons from the papers.

### PROGRAM:

```
from helper import get_openai_api_key
OPENAI_API_KEY = get_openai_api_key()
import nest_asyncio
nest_asyncio.apply()
urls = [
"https://openreview.net/pdf?id=hSyW5go0v8",
"https://openreview.net/pdf?id=VTF8yNQM66",
"https://openreview.net/pdf?id=6PmJoRfdaK"
]
papers = [
"selfrag.pdf",
"swebench.pdf",
"longlora.pdf"
]
for url, paper in zip(urls, papers):
    get_ipython().system('wget "{url}" -O "{paper}"')
from utils import get_doc_tools
from pathlib import Path
paper_to_tools_dict = {}
for paper in papers:
    print(f"Getting tools for paper: {paper}")
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]
all_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]
from llama_index.llms.openai import OpenAI
llm = OpenAI(model="gpt-3.5-turbo")
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex
obj_index = ObjectIndex.from_objects(
    all_tools,
    index_cls=VectorStoreIndex,
)
obj_retriever = obj_index.as_retriever(similarity_top_k=3)
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner
agent_worker = FunctionCallingAgentWorker.from_tools(
    tool_retriever=obj_retriever,
    llm=llm,
    system_prompt="""You are an agent designed to answer queries over a set of given papers.
Please always use the tools provided to answer a question. Do not rely on prior knowledge.""",
    verbose=True
)
agent = AgentRunner(agent_worker)
response = agent.query(
"Summarize the main idea of Self-RAG and SWE-Bench"
)
print(str(response))
response = agent.query(
"What problem does Self-RAG solve compared to traditional retrieval methods?"
)
print(str(response))
response = agent.query(
"Compare the evaluation method used in SWE-Bench with the approach used in Self-RAG"
)
print(str(response))
print("Done by Venkata Mohan N")
```

### OUTPUT:

<img width="1280" height="658" alt="image" src="https://github.com/user-attachments/assets/a6107bb7-d968-49d8-8fbd-d2b29ad35cca" />
<img width="1330" height="654" alt="image" src="https://github.com/user-attachments/assets/a1c3c5ec-9c30-4864-85a9-87de6ed679aa" />

<img width="1014" height="914" alt="image" src="https://github.com/user-attachments/assets/06b68bda-8790-4fe1-bb80-3c1a3c0498ae" />
<img width="1092" height="410" alt="image" src="https://github.com/user-attachments/assets/73f98932-a4f7-466b-9864-8bf7fa4ac04d" />


### RESULT:
