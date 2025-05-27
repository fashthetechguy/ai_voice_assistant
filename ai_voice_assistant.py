
import speech_recognition as sr
import pyttsx3
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM



# Load the Ai Model

llm = OllamaLLM(model = "mistral")


#Initialize Memory (Langchain vs 1.0+)
chat_history = ChatMessageHistory()

# Intialize Text to speech Engine
engine = pyttsx3.init()
engine.setProperty("rate" , 160) # Adjust Speaking speed


# Speech Recognition
recognizer = sr.Recognizer()

#Function to Speak
def speak(text):
	engine.say(text)
	engine.runAndWait()


#Function to Listen
def listen():
	with sr.Microphone() as source:
		print("\nListening")
		recognizer.adjust_for_ambient_noise(source)
		audio = recognizer.listen(source)
	try:
		query = recognizer.recognize_google(audio)
		print(f"you said : {query}")
		return query.lower()
	except sr.UnknownValueError:
		print("Sorry i could not understand .Try Again")
		return""
	except sr.RequestError:
		print("Speech Recognition Unavailable")
		return""

#AI Chat Prompt

prompt = PromptTemplate(
		input_variable = ["chat_history", "question"],
		template = "Previous converstation: {chat_history}\nUser: {question}\nAI:")




# Function to process AI Response

def run_chain(question):
	#Retrieve past chat history manually
	chat_history_text = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in chat_history.messages])

	# Run Ai response generation

	response = llm.invoke(prompt.format(chat_history = chat_history_text ,  question  = question))


	#Store new user input and Ai response in Memory
	chat_history.add_user_message(question)
	chat_history.add_ai_message(response)

	return response



# Main Loop

speak("Hello ! I am your AI Assistant. How can I help you")
while True:
	query = listen()
	if "exit" in query or "stop" in query:
		speak("Goodbye , Have a great day.")
		break
	if query:
		response = run_chain(query)
		print("\nAI Response: {response}")
		speak(response)