from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.schema import HumanMessage
from langchain.prompts.example_selector import LengthBasedExampleSelector

app = Flask(__name__)

# Your secret key for the Flask application. Protects against cross-site request forgery.
# You can set this in your environment or .env file.
app.secret_key = os.environ.get("FLASK_SECRET_KEY")

# Enable Cross-Origin Resource Sharing (CORS) to allow requests from different origins.
CORS(app)

# Define the route for handling POST requests to query the OpenAI model.
@app.route("/api", methods=["POST"])
def query_openai():
    # Parse the input data from the request as JSON.
    data = request.get_json()
    input_text = data.get('input')

    # Check if the input data is valid.
    if not input_text:
        return jsonify({"error": "Invalid input data"}), 400

    # Initialize the OpenAI model with your OpenAI API key.
    llm = ChatOpenAI(
        temperature=0,
        model_name='gpt-3.5-turbo',
        openai_api_key=os.environ.get("OPENAI_SECRET_KEY")
    )

    # Define a template for your prompts, which will be filled in later with examples.
    formatted_template = '''EXAMPLE PROMPT HERE {input}?'''
    prompt_tmplt = PromptTemplate(
        input_variables=["input"],
        template=formatted_template,
    )

    # Create the final prompt by replacing the {input} placeholder with the user's input.
    final_prompt = prompt_tmplt.format(input=input_text)

    # Generate a response from the OpenAI model using the final prompt.
    resp = llm([HumanMessage(content=final_prompt)])

    # Return the model's response as JSON.
    return jsonify({
        'statusCode': 200,
        'body': resp.content
    })

if __name__ == "__main__":
    # Start the Flask application in debug mode.
    app.run(debug=True)
