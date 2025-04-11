import tensorflow as tf 
import numpy as np
from tensorflow.keras.preprocessing import image
from gpt4all import GPT4All  


model = tf.keras.models.load_model("pest_disease_recognition.h5")


class_labels = [
    "Disease_Early Blight", 
    "Disease_Leaf Mold", "Pest_Aphids",
    "Pest_Corn Earworms", "Pest_Spider Mites"]

solution_model = GPT4All("mistral-7b-instruct-v0.1.Q4_0.gguf")



def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def user_prompt_for(disease_name):
    return f"""
You are an expert agricultural assistant providing information on '{disease_name}'.

Your response should be structured as follows:

# {disease_name}

**Scientific Name:** *[Scientific Name]*  
**Common Names:** [List common names]  

## 🌱 Description  
[Provide a brief description of the disease or pest]  

## 🔍 Causes & Symptoms  
- Causes: [List causes]  
- Symptoms: [List symptoms]  

## 💊 Cure & Treatment  
- Recommended Treatments: [List medications or treatments]  
- Home Remedies: [Provide natural remedies if applicable]  

## 🛡️ Prevention Tips  
- [Provide best practices for avoiding this issue]  

## 📚 Additional Information  
- [Any other useful details or expert advice]  
    """



def get_solution(pest_or_disease):
   
    system_prompt = f"""
You are an expert agricultural consultant specializing in plant health. Your task is to provide an accurate, structured, and scientifically backed solution for the detected issue: '{pest_or_disease}'.
    """

    
    user_prompt = user_prompt_for(pest_or_disease)


    final_prompt = system_prompt + "\n\n" + user_prompt

    response = solution_model.generate(final_prompt, max_tokens=500)
    return response



def predict_and_suggest_solution(img_path):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    
    print(f"\n🔍 **Detected:** {predicted_class}")
    solution = get_solution(predicted_class)
    
    print("\n💡 **Suggested Solution:**")
    print(solution)



image_path = r"C:\Users\Ayesha\Downloads\spider-mite-tetranychidae-640x480.jpg"
predict_and_suggest_solution(image_path)
