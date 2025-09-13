#CryoSensor

A project submission for the September 2025 run of the LLM Hackathon (https://llmhackathon.github.io).

This project aims to create an LLM assistant that can provide accurate information on critical chemical and physical properties of process fluids through the use of RAG.

During the hackathon, this projects was focused mainly on uses by LNG plant operators; but the use cases here can easily be extended to other settings of interest.

One of the goals of this project are to give the agent access to certain scientific tools; however, the project doesn't currently do that. The thermo library is merely imported and presented on a front end in this project, without LLM integration.

The application does currently take advantage of RAG to find and reference information in provided literature.

There needs to be a pdf document in a docs/ folder in the same directory as the main python script for the application to run.

Several dependancies are needed to run the application:
```pip install fastapi uvicorn pydantic langchain langchain-community langchain-groq langchain-chroma langchain-huggingface thermo sentence-transformers transformers torch chromadb```

We highly discourage deployment of this application, even for personal use. The code has been mostly AI generated and was not properly reviewed and evaluated for security, precision, and accuracy.

Created by:
- Suphaklit Wangdee
- Hassan Alsayoud
- Najla Albassam

Many thanks to the organizing team of the LLM Hackathon (https://llmhackathon.github.io/about) for hosting an event that sparks many great ideas for LLM applications in science and industry.
Special thanks to the organizing team at KFUPM for providing the facilities and a supportive environment that enabled us to focus and work effectively on our projects.
