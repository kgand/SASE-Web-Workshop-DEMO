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
    prompt = data.get('prompt')

    # Check if the input data is valid.
    if not prompt:
        return jsonify({"error": "Invalid input data"}), 400

    # Initialize the OpenAI model with your OpenAI API key.
    llm = ChatOpenAI(
        temperature=0,
        model_name='gpt-3.5-turbo',
        openai_api_key=os.environ.get("OPENAI_SECRET_KEY")
    )

    # Create a dynamic prompt using PromptTemplate and FewShotPromptTemplate.

    # Define a template for your prompts, which will be filled in later with examples.
    formatted_template = '''{example_query} {example_response}'''
    prompt_tmplt = PromptTemplate(
        input_variables=["example_query", "example_response"],
        template=formatted_template,
    )

    # Define your example_selector and examples here.
    # You need to populate the 'examples' variable with your data.
    examples = [...]

    # Create a prompt selector based on the length of examples.
    prompt_selector = LengthBasedExampleSelector(
        examples=examples,
        example_prompt=prompt_tmplt
    )

    # Create a dynamic prompt using examples and user's input.
    dynamic_prompt = FewShotPromptTemplate(
        example_selector=prompt_selector,
        example_prompt=prompt_tmplt,
        prefix="""Can you return an array of objects as a JSON formatted string that are geographically relevant to an arbitrary query?

        REQUIREMENTS:
        - Each object in the array should contain 3 keys: lon, lat, blurb
        - lon is the longitude of the coords for each match to the query
        - lat is the latitude of the coords for each match to the query
        - blurb is the 1-3 sentence answer to the query along with information about the environmental concerns of the city or region in which the coords exist
        - The array should be max length 3 items
        - the overall length of the answer should be maximum 500 characters and should be a fully parsable JSON string
        - if you cannot provide accurate information then please provide your best guess along with a disclaimer
        """,
        suffix="Here is the arbitrary query...\n\n{input}\n",
        input_variables=["input"],
        example_separator="\n\n",
    )

    # Create the final prompt by replacing the {input} placeholder with the user's input.
    final_prompt = dynamic_prompt.format(input=prompt)

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
