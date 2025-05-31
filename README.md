# Maibloom AIcore

Mai Bloom AI-core is a sophisticated artificial intelligence engine built upon advanced AI modules provided by Deepset and HuggingFace, and it is further enriched by DuckDuckGo’s comprehensive knowledge base. This innovative integration offers intelligent and tailored solutions designed to address the ever-evolving challenges in modern technology. 

---

## Theory and Logic

HuggingFace serves as the foundation for this model, where all AI models are executed. Deepset's `roberta-base-squad2` is employed as the question-answering model, providing responses to user queries. You can check out the project on HuggingFace: [Deepset's roberta-base-squad2](https://huggingface.co/deepset/roberta-base-squad2).

Additionally, DuckDuckGo's Python libraries help retrieve fresh information from the web related to the query, supplying new data for the Deepset question-answering model.

---

## Installation

> [!CAUTION]
> This AI model may require about 5 to 6 gigabytes of storage space on your root (`/`) partition!

### Method 1: Using OmniPkg Package Manager

1. **Install OmniPkg:**  
   Please follow the instructions on the OmniPkg main page: [OmniPkg on GitHub](https://github.com/maibloom/omnipkg-app)

2. **Install Maibloom AIcore:**
```
sudo omnipkg put install maibloom-aicore
```

### Method 2: using git

you can git this project and run the bash script file:

```
git clone https://www.github.com/maibloom/maibloom-aicore/

cd maibloom-aicore

chmod +x *

sudo bash install.sh
```