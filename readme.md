
# GPT Response Generator with Defense Against Adversarial Prompts
This project leverages OpenAI's GPT models to generate intelligent and creative responses to user input, with a specific focus on defending against "adversarial prompts." Adversarial prompts can be used to manipulate the system into generating undesired content, and this Python script attempts to alleviate this concern by detecting potential objectionable content.

## Features
- **Response Generation:** Utilizes the GPT models to create intuitive responses.
- **Defense Against Adversarial Prompts:** Employs a system to check the first 30 words of a response to determine whether the content is objectionable, as part of the system's defense mechanism.
- **Concurrency Support:** Makes use of ThreadPoolExecutor for efficient parallel processing.
- **Streamed Output:** Allows real-time monitoring of response generation.

## Requirements
- Python 3.6 or higher
- OpenAI Python package
- termcolor package

## Usage
Simply run the main script:
```bash
python main.py
```
Then, enter your text when prompted. The script will generate a response and perform a check to ensure that it's not objectionable, safeguarding against adversarial attempts to manipulate the response.

## Understanding the Code
- `check_objectionable_response`: A function that communicates with the GPT model to check for objectionable content and prevent adversarial manipulation.
- `regular_response`: A function that gets the regular response from the GPT model.
- `main`: The main function that coordinates both tasks and provides the final output.

## Additional Resources
- **YouTube Channel:** Learn how to build AI-powered apps with GPT on [EchoHive's YouTube Channel](https://www.youtube.com/@echohive).
- **Website:** Search over videos and find code download links at [EchoHive Live](https://www.echohive.live/).
- **LLM Paper Summarizer:** Summarizes every new paper related to "large language models" from arXiv automatically every hour. Visit [LLM Papers](https://llmpapers.up.railway.app/).
- **Support:** If you want to support the efforts, you can do so through [Patreon](https://www.patreon.com/echohive42).

## Contributions
Feel free to open issues or submit pull requests if you want to enhance this project. Your insights and contributions are welcome!
