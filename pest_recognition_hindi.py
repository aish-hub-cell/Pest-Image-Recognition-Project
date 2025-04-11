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
आप एक विशेषज्ञ कृषि सहायक हैं जो '{disease_name}' के बारे में जानकारी प्रदान कर रहे हैं।

आपका उत्तर निम्नलिखित संरचना में होना चाहिए और हिंदी में होना चाहिए:

# {disease_name}

**वैज्ञानिक नाम:** *[वैज्ञानिक नाम]*  
**सामान्य नाम:** [सामान्य नामों की सूची]  

## 🌱 विवरण  
[रोग या कीट का संक्षिप्त विवरण दें]  

## 🔍 कारण और लक्षण  
- कारण: [कारणों की सूची दें]  
- लक्षण: [लक्षणों की सूची दें]  

## 💊 उपचार और समाधान  
- अनुशंसित उपचार: [दवाओं या उपायों की सूची दें]  
- घरेलू उपचार: [प्राकृतिक उपचार प्रदान करें यदि लागू हो]  

## 🛡️ रोकथाम के उपाय  
- [इस समस्या से बचने के सर्वोत्तम तरीके प्रदान करें]  

## 📚 अतिरिक्त जानकारी  
- [कोई अन्य उपयोगी विवरण या विशेषज्ञ सलाह]  
    """


def get_solution(pest_or_disease):
   
    system_prompt = f"""
आप एक कृषि विशेषज्ञ हैं जो पौधों के स्वास्थ्य में विशेषज्ञता रखते हैं। 
आपका कार्य '{pest_or_disease}' के लिए एक सटीक, संरचित और वैज्ञानिक समाधान हिंदी में प्रदान करना है।
"""

    
    user_prompt = user_prompt_for(pest_or_disease)


    final_prompt = system_prompt + "\n\n" + user_prompt

    response = solution_model.generate(final_prompt, max_tokens=500)
    return response



def predict_and_suggest_solution(img_path):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    
    print(f"\n🔍 **पहचाना गया समस्या:** {predicted_class}")
    solution = get_solution(predicted_class)
    
    print("\n💡 **सुझाया गया समाधान:**")
    print(solution)



image_path = r"C:\Users\Ayesha\Downloads\spider-mite-tetranychidae-640x480.jpg"
predict_and_suggest_solution(image_path)
