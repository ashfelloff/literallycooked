import cv2
import numpy as np
from anthropic import Anthropic
import time
from typing import List, Dict
import json
import os
from dotenv import load_dotenv, find_dotenv
import io
import base64
import requests
import platform
import tempfile
import subprocess

# Add debug prints for dotenv
print(f"Current working directory: {os.getcwd()}")
env_path = find_dotenv()
print(f"Found .env file at: {env_path}")

# Try to load with explicit path
load_dotenv(env_path)

class LiterallyCookedApp:
    def __init__(self):
        # Debug prints
        print("Loading environment variables...")
        print(f"ANTHROPIC exists: {'ANTHROPIC' in os.environ}")
        api_key = os.getenv('ANTHROPIC', 'NOT_FOUND')
        print(f"Full API key length: {len(api_key)}")
        print(f"First 10 chars of key: {api_key[:10]}")
        
        # Core initialization
        self.anthropic = Anthropic(api_key=os.getenv('ANTHROPIC'))
        self.api_key = os.getenv('GOOGLE_CLOUD_API_KEY')
        self.vision_base_url = "https://vision.googleapis.com/v1/images:annotate"
        self.cap = cv2.VideoCapture(0)
        
        # State variables
        self.detected_items: List[str] = []
        self.current_recipe: Dict = {}
        self.last_detection_time = time.time()
        self.detection_cooldown = 1.0
        self.last_center = None
        self.box_size = 300
        self.smoothing = 0.7
        self.current_detection = None
        self.countdown_start = None
        self.countdown_duration = 15
        self.recipe_generated = False

    def detect_foods(self, frame) -> List[str]:
        frame_height, frame_width = frame.shape[:2]
        
        # Convert the frame to jpg and then to base64
        success, encoded_image = cv2.imencode('.jpg', frame)
        image_content = encoded_image.tobytes()
        image_b64 = base64.b64encode(image_content).decode('utf-8')
        
        # Prepare the request
        request_json = {
            "requests": [
                {
                    "image": {
                        "content": image_b64
                    },
                    "features": [
                        {
                            "type": "OBJECT_LOCALIZATION",
                            "maxResults": 10
                        }
                    ]
                }
            ]
        }
        
        # Make the API request
        response = requests.post(
            f"{self.vision_base_url}?key={self.api_key}",
            json=request_json
        )
        
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            return []
        
        results = response.json()
        
        # Process detections
        detected = []
        best_detection = None
        highest_confidence = 0
        
        print("\nFood detections:")
        for annotation in results.get('responses', [{}])[0].get('localizedObjectAnnotations', []):
            # Check if it's a food item
            if annotation['name'].lower() in ['food', 'fruit', 'vegetable', 'beverage', 'apple', 'banana', 'orange']:
                confidence = annotation['score']
                print(f"{annotation['name']}: {confidence:.2f}")
                
                try:
                    if confidence > highest_confidence:
                        highest_confidence = confidence
                        
                        # Get bounding box - with error handling
                        vertices = annotation['boundingPoly']['normalizedVertices']
                        if len(vertices) >= 4:  # Make sure we have all corners
                            x1 = int(vertices[0].get('x', 0) * frame_width)
                            y1 = int(vertices[0].get('y', 0) * frame_height)
                            x2 = int(vertices[2].get('x', 0) * frame_width)
                            y2 = int(vertices[2].get('y', 0) * frame_height)
                            
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            
                            best_detection = (annotation['name'], confidence, (center_x, center_y))
                except (KeyError, IndexError) as e:
                    print(f"Warning: Could not process bounding box for {annotation['name']}")
                    continue
                    
                if annotation['name'] not in detected:
                    detected.append(annotation['name'])
        
        if best_detection:
            self.current_detection = best_detection
        
        return detected

    def open_recipe_terminal(self, recipe: Dict):
        # Create HTML content with some basic styling
        html_content = f"""
        <html>
        <head>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 40px auto;
                    padding: 20px;
                    line-height: 1.6;
                    background-color: #f5f5f5;
                }}
                h1 {{
                    color: #2c3e50;
                    text-align: center;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                }}
                .section {{
                    background: white;
                    padding: 20px;
                    margin: 20px 0;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                ul {{
                    list-style-type: none;
                    padding-left: 20px;
                }}
                li {{
                    margin: 10px 0;
                    position: relative;
                }}
                li:before {{
                    content: "•";
                    color: #3498db;
                    font-weight: bold;
                    position: absolute;
                    left: -20px;
                }}
                .improvements {{
                    background: #e8f4f8;
                    padding: 15px;
                    border-radius: 5px;
                }}
            </style>
        </head>
        <body>
            <h1>🍳 {recipe.get('name', '').upper()}</h1>
            
            <div class="section">
                <h2>📝 Ingredients</h2>
                <ul>
                    {''.join(f'<li>{ingredient}</li>' for ingredient in recipe.get('ingredients', []))}
                </ul>
            </div>
            
            <div class="section">
                <h2>🔧 Tools Needed</h2>
                <ul>
                    {''.join(f'<li>{tool}</li>' for tool in recipe.get('tools_needed', []))}
                </ul>
            </div>
            
            <div class="section">
                <h2>👩‍🍳 Instructions</h2>
                <ol>
                    {''.join(f'<li>{step}</li>' for step in recipe.get('instructions', []))}
                </ol>
            </div>
            
            <div class="section">
                <h2>💡 Potential Improvements</h2>
                <div class="improvements">
                    {recipe.get('improvements', '')}
                </div>
            </div>
            <h3>by <a href="https://github.com/ashfelloff">ashfelloff</a></h3>
        </body>
        </html>
        """
        
        # Create a temporary HTML file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.html') as f:
            f.write(html_content)
            recipe_file = f.name
        
        # Open the HTML file in the default web browser
        if platform.system() == "Darwin":  # macOS
            subprocess.run(['open', recipe_file])
        elif platform.system() == "Linux":
            subprocess.run(['xdg-open', recipe_file])
        elif platform.system() == "Windows":
            os.startfile(recipe_file)
        
        
        time.sleep(2)

    def get_recipe_suggestion(self, ingredients: List[str]) -> Dict:
        print(f"Generating recipe for ingredients: {ingredients}")
        
        prompt = f"""Given these ingredients: {', '.join(ingredients)}, and assuming basic kitchen tools and cutlery,
        suggest a creative recipe that uses them. 
        Then, suggest potential improvements or variations if the cook had additional ingredients.
        You must respond with ONLY valid JSON in this exact format:
        {{
            "name": "Recipe Name",
            "ingredients": ["ingredient 1 with quantity", "ingredient 2 with quantity"],
            "tools_needed": ["tool 1", "tool 2"],
            "instructions": ["step 1", "step 2", "step 3"],
            "improvements": "Suggestions for additional ingredients and variations"
        }}"""
        
        response = self.anthropic.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        try:
            content = response.content[0].text if isinstance(response.content, list) else response.content
            print("Received response from Claude:")
            print(content)  # Add this debug print
            print("\nTrying to parse as JSON...")
            recipe = json.loads(content)
            print("Successfully parsed JSON")
            self.open_recipe_terminal(recipe)
            return recipe
        except (json.JSONDecodeError, IndexError, AttributeError) as e:
            print(f"Error parsing recipe: {e}")
            return {
                "name": f"Simple {ingredients[0]} Recipe",
                "ingredients": ingredients,
                "tools_needed": ["Basic kitchen tools"],
                "instructions": ["Prepare ingredients as desired"],
                "improvements": "Add seasonings to taste"
            }

    def draw_ui(self, frame, detected_items: List[str]):
        
        cv2.putText(frame, "Detected Foods:", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        
        if self.current_recipe:
            recipe_y = 60 + (len(detected_items) * 30) + 20
            recipe = self.current_recipe
            
            
            cv2.putText(frame, f"Recipe: {recipe.get('name', '')}", 
                       (10, recipe_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            
            if 'nutrition' in recipe:
                cv2.putText(frame, f"Nutrition: {recipe['nutrition']}", 
                           (10, recipe_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        
        cv2.putText(frame, "Press 'r' for recipe | 's' to save | 'q' to quit", 
                   (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def save_recipe(self):
        if self.current_recipe:
            filename = f"recipe_{int(time.time())}.json"
            with open(filename, 'w') as f:
                json.dump(self.current_recipe, f, indent=4)
            return filename
        return None

    def draw_box(self, frame):
        if not self.current_detection:
            return
        
        frame_height, frame_width = frame.shape[:2]
        name, confidence, (center_x, center_y) = self.current_detection
        
        if self.last_center is None:
            self.last_center = (center_x, center_y)
        
        new_x = int(self.last_center[0] * self.smoothing + center_x * (1 - self.smoothing))
        new_y = int(self.last_center[1] * self.smoothing + center_y * (1 - self.smoothing))
        
        new_x = max(self.box_size // 2, min(frame_width - self.box_size // 2, new_x))
        new_y = max(self.box_size // 2, min(frame_height - self.box_size // 2, new_y))
        
        x = new_x - self.box_size // 2
        y = new_y - self.box_size // 2
        
        self.last_center = (new_x, new_y)
        
        cv2.rectangle(frame, (x, y), (x + self.box_size, y + self.box_size), (0, 255, 0), 2)
        
        label_text = f"{name}: {confidence:.2f}"
        cv2.putText(frame, label_text,
                  (x, y - 10),
                  cv2.FONT_HERSHEY_SIMPLEX,
                  0.6, (0, 255, 0), 2)
        
        cv2.line(frame, (new_x - 10, new_y), (new_x + 10, new_y), (0, 255, 0), 1)
        cv2.line(frame, (new_x, new_y - 10), (new_x, new_y + 10), (0, 255, 0), 1)

    def draw_countdown(self, frame):
        if self.countdown_start is None:
            self.countdown_start = time.time()

        elapsed_time = time.time() - self.countdown_start
        remaining_time = max(0, self.countdown_duration - elapsed_time)
        
        
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
        
        
        countdown_text = f"Generating recipe in: {int(remaining_time)} seconds"
        cv2.putText(frame, countdown_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        
        if remaining_time == 0 and not self.recipe_generated:
            self.current_recipe = self.get_recipe_suggestion(self.detected_items)
            self.recipe_generated = True

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            current_time = time.time()
            
            
            if current_time - self.last_detection_time >= self.detection_cooldown:
                self.detected_items = self.detect_foods(frame)
                self.last_detection_time = current_time
            
            self.draw_box(frame)  
            self.draw_countdown(frame)  
            
            cv2.imshow("LiterallyCooked", frame)
            
            
            if self.recipe_generated:
                time.sleep(2)  
                break
            
            if cv2.waitKey(1) & 0xFF == 27:  
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = LiterallyCookedApp()
    app.run()
