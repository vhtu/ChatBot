from chatterbot.trainers import ListTrainer
from chatterbot import ChatBot

bot = ChatBot('Test')

conv = open('chat.txt','r').readlines()

bot.set_trainer(ListTrainer)

bot.train(conv)

while true:
	resquest = input('You: ')
	response = bot.get_response(resquest)
	print('Bot: ', response)