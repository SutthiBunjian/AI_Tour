import torch
import gradio as gr
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForCausalLM
import requests
import json
import time

#####################################################################
# Load API config from JSON file
def load_api_config(json_file='src/utils/api_config.json'):
    with open(json_file, 'r') as file:
        configs_list = json.load(file)
        configs_dict = {}
        for config in configs_list:
            model_name = config.get('modelName')
            if model_name:
                configs_dict[model_name] = config
        
    return configs_dict, configs_list

def load_local_api_config(configs):
    localconfigs = {}
    localAPIs = []
    for config in configs:
        if config['apiKey'] == 'local':
            localAPIs.append(config['modelName'])
            config['tokenizer'] = AutoTokenizer.from_pretrained(config['basicUrl'])
            config['model'] = AutoModelForCausalLM.from_pretrained(config['basicUrl'])
            localconfigs[config['modelName']] = config
            # print(f"Loaded tokenizer for model: {config['modelName']}")
    return localconfigs, localAPIs

# Load user presets from JSON file
def load_user_presets(json_file='data/user_presets.json'):
        with open(json_file, 'r') as file:
            presets_list = json.load(file)
            presets_dict = {}
            for preset in presets_list:
                preset_name = preset.get('preset_name')
                if preset_name:
                    presets_dict[preset_name] = preset
        
        return presets_dict, presets_list
#####################################################################
# Initialize settings
user_needs_input_lists = ['scientific name', 'habitat information', 'diet details', 'fun facts','unique characteristics', 'conservation status', 'ecological role', 'behavior', 'cultural significance']

# Load user presets
presets_file = 'data\\user_presets.json'
presets_dict, presets_list = load_user_presets(presets_file )
preset_names = list(presets_dict.keys()) if presets_dict else []
# print(f"\nLoaded {len(presets_list)} user presets:")

#####################################################################

# Function to classify animal using CLIP and get statistics
def animal_classifier(image):
    if image is None:
        return "Please upload an image.", "", 0.0, []
    
    prompts = [f"a photo of a {animal}" for animal in animals]
    
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        inputs = CLIP_processor(text=prompts, images=image, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = CLIP_model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)

        top_probs, top_idxs = torch.topk(probs[0], 3)
        animal_results_name = []
        classification_results = []
        
        for i in range(3):
            top_animal = animals[top_idxs[i]]
            confidence = top_probs[i].item()
            classification_results.append(f"{i+1}. {top_animal}, Confidence: {confidence:.3f}")
            animal_results_name.append(top_animal)
            
        top3_classification_results = "Top 3 predictions:\n" + "\n".join(classification_results)
        top1_animal = animals[top_idxs[0]]
        top1_confidence = top_probs[0].item()
        
        return top3_classification_results, top1_animal, top1_confidence, animal_results_name
        
    except Exception as e:
        error_msg = f"Classification Error: {str(e)}"
        print(error_msg)
        return error_msg, "", 0.0, []
    
# Function to create prompt and get description
def get_animal_description(role, animal_name, modelName, user_needs=None):
    
    if animal_name is None or animal_name == "":
        return "No animal selected for description."
    
    # incorporate it into the prompt
    needs_text = ""
    if user_needs and isinstance(user_needs, list) and len(user_needs) > 0:
        needs_text = "\nMy specific interests:\n" + "\n".join([f"- {need}" for need in user_needs])

    prompt = f"Describe the {animal_name} from the perspective of a {role}.{needs_text}"
    prompt += f"\nMake the description interesting and concise appropriate for a {role}.\n\
Make it into different sections with headings.\n\
Formatting: please do not include ##, **, and #### in the response. Use plain text only, less than 200 words."

    print(f"Generated Prompt:\n {prompt}") 
    return submit(prompt, modelName)

# Function to interact with LLM API
def submit(message, modelName):
    if modelName in localAPIs:
        messages = [{"role": "user", "content": message}]
        
        tokenizer = local_configs[modelName]['tokenizer']
        model = local_configs[modelName]['model']   

        input_time = time.time()
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        tokenize_time = time.time() - input_time
        print(f"Tokenization time: {tokenize_time:.2f} seconds")

        generate_start = time.time()
        outputs = model.generate(
            **inputs, 
            max_new_tokens=400,  
            temperature=0.7,     
            pad_token_id=tokenizer.eos_token_id
        )
        generate_time = time.time() - generate_start
        print(f"Generation time: {generate_time:.2f} seconds")
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response
        
    else:
        config = configs_dict[modelName]
        apiVersion = config['apiVersion']
        modelName = config['modelName']
        baseURL = config['basicUrl']
        print(f"Using {modelName} API for response.")
        
        conversation = [{"role": "user", "content": message}]
        url = baseURL + "/deployments/" + modelName + "/chat/completions?api-version=" + apiVersion
        headers = {'Content-Type': 'application/json', 'api-key': config['apiKey']}
        payload = {'messages': conversation}
        
        try:
            generate_start = time.time()
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                assistant_reply = data["choices"][0]["message"]["content"]
                generate_time = time.time() - generate_start
                print(f"API Generation time: {generate_time:.2f} seconds")
                return assistant_reply
            else:
                return f"Error: {response.status_code} - {response.text}"
        except Exception as e:
            return f"API Error: {str(e)}"


# Function to apply selected preset
def show_preset(preset_name):
    """Update role and show preset details when a preset is selected"""
    if preset_name and preset_name in presets_dict:
        preset = presets_dict[preset_name]
        user_role = preset.get('role', 'Not specified')
        user_needs = preset.get('needs', [])
        needs_text = "\n".join([f"• {need}" for need in user_needs]) if user_needs else "No specific needs"
        user_model = preset.get('model')
        return f"Preset: {preset_name}\n\nYour role: {user_role}\n\nYour interests:\n{needs_text}\n\nModel: {user_model}"
    return "Select a preset"
# Process function for Gradio
def process_with_description(image, preset_name=None, role_input=None, modelName_input=None, user_needs_input=None):
    # Process classification
    top3_classification_results, top1_animal, top1_confidence, animal_results_name = animal_classifier(image)
    
    if not top3_classification_results.startswith("Classification Error"):
        # Initialize variables
        user_role = "Zoo visitor"  # Default role
        modelName = "Qwen3-0.6B"   # Default model
        user_needs = []
        
        # Manual mode
        if role_input is not None and modelName_input is not None and user_needs_input is not None:
            # Manual mode
            user_role = role_input
            modelName = modelName_input
            user_needs = user_needs_input if user_needs_input else []
            print(f"Using manual settings - Role: {user_role}, Model: {modelName}, User Needs: {user_needs}")
        
        # Preset mode
        elif preset_name and preset_name in presets_dict:
            preset = presets_dict[preset_name]
            user_role = preset.get('role', user_role)
            user_needs = preset.get('needs', [])
            modelName = preset.get('model', modelName)
            print(f"Applying preset: {preset_name} with role: {user_role}, model: {modelName}, and user needs: {user_needs}")
        
        # Get description using the appropriate values
        description = get_animal_description(user_role, top1_animal, modelName, user_needs)

        # Create dropdown with safe indexing
        choices = animal_results_name if animal_results_name else []
        default_value = choices[0] if choices else None
        
        return top3_classification_results, description, gr.Dropdown(choices=choices, value=default_value)
    else:
        return top3_classification_results, "Cannot get description due to classification error", gr.Dropdown(choices=[], value=None)
def save_preset(preset_name, role, modelName, user_needs):
    global presets_dict, presets_list, preset_names, presets_file
    with open(presets_file, 'r') as file:
        presets_list = json.load(file)

    preset = {
        "preset_name": preset_name,
        "role": role,
        "needs": user_needs,
        "model": modelName
    }

    presets_list.append(preset)
    with open(presets_file, 'w') as file:
        json.dump(presets_list, file, indent=2)
    
    presets_dict, presets_list = load_user_presets(presets_file)
    preset_names = list(presets_dict.keys()) if presets_dict else []

# Create Gradio interface
with gr.Blocks(title="Smart Animal Classifier") as demo:
    gr.Markdown("# 🐾 Smart Animal Classifier")
    gr.Markdown("Upload an animal photo to classify it and get a tailored description!")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Animal Photo")
            
            # Preset selection
            preset_input = gr.Dropdown(
                label="Select User Preset",
                choices=preset_names,
                value=preset_names[0] if preset_names else None,
                interactive=True,
                info="Choose a user preset with predefined interests"
            )
            
            # Display preset details
            preset_details = gr.Markdown("Select a preset to see details")
            
            # Checkbox to show/hide manual controls
            manual_role_checkbox = gr.Checkbox(
                label="Show Manual Controls", 
                value=False,
                info="Check to manually override role and model settings"
            )
            
            # Container for manual controls (initially hidden)
            with gr.Column(visible=False) as manual_controls:
                gr.Markdown("### Manual Settings")
                
                preset_name_input = gr.Textbox(label="Custom Preset Name", placeholder="Enter a name for this preset", interactive=True)
                # Manual role selection
                role_input = gr.Dropdown(
                    label="Role Selection",
                    choices=['Zoo visitor', 'Student', 'Researcher', 'Tourist', 'Child'],
                    value='Zoo visitor',
                    interactive=True,
                    info="Choose a role to tailor the description"
                )
                
                modelName_input = gr.Radio(
                    choices=["Qwen3-0.6B", "Qwen2.5-0.5B", "deepseek-v3", "gpt-4.1-mini", "gemini-2.5-flash"],
                    label="Processing Mode",
                    value="deepseek-v3",
                    info="Choose between local model or cloud API"
                )
                user_needs_input = gr.CheckboxGroup(
                    label="Select Your Interests",
                    choices=user_needs_input_lists,
                    interactive=True,
                    info="Select specific interests to include in the description"
                )
                save_btn = gr.Button("Save Current Settings as Preset", variant="primary")
                manual_classify_btn = gr.Button("Classify & Get Description with Manual Settings", variant="secondary")

            classify_btn = gr.Button("Classify & Get Description", variant="primary")
        
        with gr.Column():
            classification_output = gr.Textbox(label="Classification Results", lines=4)
            description_output = gr.Textbox(label="Animal Description", lines=8)

            animal_selector = gr.Dropdown(
                label="Not the right animal? Select from top predictions:",
                choices=[],
                interactive=True,
                info="Choose a different animal from the top predictions"
            )
            update_btn = gr.Button("Get Description for Selected Animal", variant="secondary")
    
    # Event handlers
    preset_input.change(
        fn=show_preset,
        inputs=[preset_input],
        outputs=[preset_details]
    )
    
    # Show/hide manual controls based on checkbox
    manual_role_checkbox.change(
        fn=lambda x: gr.Column(visible=x),
        inputs=[manual_role_checkbox],
        outputs=[manual_controls]
    )
    manual_classify_btn.click(
        fn=process_with_description,
        inputs=[image_input, preset_name_input, role_input, modelName_input, user_needs_input],
        outputs=[classification_output, description_output, animal_selector]
    )
    classify_btn.click(
    fn=process_with_description,
    inputs=[image_input, preset_input],
    outputs=[classification_output, description_output, animal_selector]
)
    
    save_btn.click(
        fn=save_preset,
        inputs=[preset_name_input, role_input, modelName_input, user_needs_input],
        outputs=[preset_details]
    ).then(
        fn=lambda: gr.Dropdown(choices=preset_names, value=preset_names[0] if preset_names else None),
        outputs=[preset_input]
    )

    update_btn.click(
        fn=get_animal_description,
        inputs=[preset_input, animal_selector, modelName_input],
        outputs=[description_output]
    )
########################################################################

# Load configurations
configs_dict, configs_list = load_api_config()
local_configs, localAPIs = load_local_api_config(configs_list)

# print(f"\nTotal API configurations loaded: {len(configs_list)}")
# print(f"Local API models available: {localAPIs}")



# Load CLIP model for animal classification
CLIP_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
CLIP_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

with open("data/animals/species_name.txt", 'r') as file:
    animals = [line.strip() for line in file if line.strip()]

# print(f"\nLoaded {len(animals)} animal species for classification.\n")
# Launch the app
if __name__ == "__main__":
    demo.launch(share=False)
