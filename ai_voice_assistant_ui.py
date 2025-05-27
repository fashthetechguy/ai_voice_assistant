import streamlit as st
import speech_recognition as sr
import pyttsx3
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM




# Load the Ai Model
llm = OllamaLLM(model = "mistral")

#Initialize Memory (Langchain vs 1.0+)
if "chat_history" not in st.session_state:
	st.session_state.chat_history = ChatMessageHistory() # store User Ai Connversation



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
		st.write("\nListening")
		recognizer.adjust_for_ambient_noise(source)
		audio = recognizer.listen(source)
	try:
		query = recognizer.recognize_google(audio)
		st.write(f"you said : {query}")
		return query.lower()
	except sr.UnknownValueError:
		st.write("Sorry i could not understand .Try Again")
		return""
	except sr.RequestError:
		st.write("Speech Recognition Unavailable")
		return""



#AI Chat Prompt

prompt = PromptTemplate(
		input_variable = ["chat_history", "question"],
		template = "Previous converstation: {chat_history}\nUser: {question}\nAI:")


def run_chain(question):
	#Retrieve past chat history manually
	chat_history_text = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in chat_history.messages])

	# Run Ai response generation

	response = llm.invoke(prompt.format(chat_history = chat_history_text ,  question  = question))


	#Store new user input and Ai response in Memory
	st.session_state.chat_history.add_user_message(question)
	st.session_state.chat_history.add_ai_message(response)

	return response



#Streamlit Web Ui

st.title("Ai Voice Assistant (Web Ui)")
st.write("Click the button below to spleak to your AI Assistant")


#Button to record Voice Input
if st.button("Start Listening"):
	user_query = listen()
	if user_query:
		ai_response = run_chain(user_query)
		st.write(f"**You:** {user_query}")
		st.write(f"**AI:** {ai_response}")
		speak(ai_response) #AI speak the response


# Display Full chat History
st.subheader("Chat History")
for msg in st.session_state.chat_history.messages:
	st.write(f"**{msg.type.capitalize()}**: {msg.content}")