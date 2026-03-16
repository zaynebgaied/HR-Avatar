# HR Avatar — Interactive HR Interview System

HR Avatar is an application that simulates a real-time job interview. A conversational avatar asks the candidate questions, transcribes their voice responses, generates natural replies, and produces an evaluation report at the end of the session.

The project supports three languages: French, English, and Arabic.

---

## What the application does

- Conducts a structured interview across 6 phases (introduction, technical skills, soft skills...)
- Transcribes the candidate's voice in real time (Faster-Whisper)
- Responds with a natural synthetic voice (Edge-TTS)
- Uses the candidate's CV and job offer to personalize questions (RAG + Ollama)
- Generates an automatic scoring report with a final hiring verdict

---

## Requirements

Before running the project, you need:

- [Docker Desktop](https://www.docker.com/products/docker-desktop)
- [Ollama](https://ollama.com/download)

Once Ollama is installed, pull the model used by the application:

```bash
ollama pull qwen2.5
```

---

## Getting started

No need to clone the entire repository. Just download the [`docker-compose.share.yml`](./docker-compose.share.yml) file, then run the following command in the folder where it's located:

```bash
docker-compose -f docker-compose.share.yml up
```

The Docker image will be pulled automatically from Docker Hub. Once the app is running, open your browser at:

```
http://localhost:8000
```

---

## Supported languages

| Language | Voice used             |
|----------|------------------------|
| French   | fr-FR-DeniseNeural     |
| English  | en-US-JennyNeural      |
| Arabic   | ar-SA-ZariyahNeural    |

---

## Interview phases

The interview is structured into 6 phases, scored out of 100 points total:

| Phase | Points |
|-------|--------|
| Introduction & Background | 10 |
| Project Relevance | 20 |
| Technical Skills | 40 |
| Soft Skills & Culture Fit | 20 |
| Curiosity & Motivation | 10 |
| Closing | — |

The final report returns one of three verdicts: **HIRE**, **TO REVIEW**, or **REJECT**.

---

## Troubleshooting

**Ollama not reachable from the Docker container?**

On Linux, update the environment variable in `docker-compose.share.yml`:

```yaml
OLLAMA_HOST=http://172.17.0.1:11434
```

