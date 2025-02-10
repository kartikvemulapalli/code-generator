import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*Tried to instantiate class '__path__._path'.*")

import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from time import time

st.title("🚀 DeepSeek Coder Code Generator 🚀")
st.write("Generate code snippets by selecting a programming language and entering a prompt.")

languages = ["Python", "JavaScript", "Java", "C++", "SQL", "HTML/CSS", "Ruby", "Go", "Rust", "PHP"]
selected_language = st.selectbox("Select a programming language:", languages)

prompt = st.text_area("Enter your prompt:", f"Write a {selected_language} function to ")

@st.cache_resource
def load_model():
    model_name = "deepseek-ai/deepseek-coder-1.3b-base"
    
    try:
        from accelerate import dispatch_model  # Ensure accelerate is installed
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        return tokenizer, model.to(device)
    
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


tokenizer, model = load_model()

if st.button("Generate Code"):
    start_time = time()
    with st.spinner("Generating code..."):
        try:
            full_prompt = f"Write a {selected_language} function: {prompt}"
            inputs = tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True).to("cuda" if torch.cuda.is_available() else "cpu")
            
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.get("attention_mask"),
                max_length=150,  # Reduce max length for faster response
                temperature=0.6,  # Lower temperature for more deterministic output
                top_p=0.8,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

            generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            st.subheader("Generated Code:")
            st.code(generated_code, language=selected_language.lower())

            end_time = time()
            st.success(f"Code generated in {end_time - start_time:.2f} seconds!")
        except Exception as e:
            st.error(f"An error occurred: {e}")
