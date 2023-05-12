#!/usr/bin/env python


from gpt_index import GPTMultiverseIndex
from gpt_index.readers.schema.base import Document

from dotenv import load_dotenv

load_dotenv()

doc = Document("This is a conversation about shoes.")
index = GPTMultiverseIndex(documents=[], generate_embeddings=True)
index.extend(doc)

doc = Document("I like welligton boots.")
index.insert(doc)

doc = Document("I like running shoes.")
index.insert(doc)

# Checkout the second reply
index.checkout(2)

doc = Document("What is so great about running shoes?")
index.insert(doc)
doc = Document("I also like running shoes!")
# Extend means insert and checkout
index.extend(doc)

index.tag('second')
index.checkout(1)
index.tag('first')

doc = Document("Oh yeah welly boots are great.")
# Extend means insert and checkout
index.extend(doc)

doc = Document("Do you like any particular brand?")
index.insert(doc)

doc = Document("Right? I wear mine every day.")
index.extend(doc)
index.tag('first')

print(index.index_struct)
print(index)

index.save_to_disk("test_index.json")

# print(index.index_struct)
# print(index)
# print(index.index_struct.get_full_repr())
#
# index.checkout(6)
# doc = Document("Yeah, I think they are called hunters? Is that a brand?")
# index.insert(doc)
# doc = Document("I just wear the regular ones from the store.")
# index.extend(doc)
#
# doc = Document("Thats fair, I also think regular ones are great.")
# index.extend(doc)

# doc = Document("This is a conversation about lambda calculus.")
# index.new(doc)
#
# doc = Document("Oh awesome, I love lambda calculus.")
# index.extend(doc)
#
# index.tag('lambda')
# index.checkout('first')


# query = index.query("What is a good brand of boot?")
# index.embeddings()
# index.get_node_similarities("I like running shoes more than boots.")


# index.checkout('lambda')
# index.clear_node_similarities()
# index.generate_summaries = True
# index.cache_size = 2
#
# doc = Document("Great! Lets get started.")
# index.extend(doc)
#
# doc = Document("So, my name is Frank and I am a tutor of lambda calculus.")
# index.extend(doc)
#

# doc = Document("Nice to meet you frank! My name is Susan. I guess I'm your student now ;)")
# index.extend(doc)


# index.generate_summaries = True
# index.cache_size = 2
#
# doc = Document("Nice to meet you Susan.")
# index.extend(doc)
#
# index.save_to_disk("test_index.json")

# doc = Document("Lets get started. Can you please remember the following: lambda calculus was invented by Alonzo Church. It is often contrasted with whatever Alan Turing was up to.")
# index.extend(doc)
# index.save_to_disk("test_index.json")


# doc = Document("Sure! I can remember that! Btw, how much do you charge for tutoring? I forgot what the advert said.")
# index.extend(doc)
#
# doc = Document("I charge $100 per hour.")
# index.extend(doc)
#
# doc = Document("Oh, that's a bit expensive. Better learn fast!")
# index.extend(doc)
#
# doc = Document("Right. So, who is Alan Turing?")
# index.extend(doc)
#
# print(index.index_struct._get_repr(summaries=True))
#
# doc = Document("Um, the guy who invented lambda calculus?")
# index.extend(doc)
#
# doc = Document("Nope! He invented the Turing machine. Better remember better, this is costing you.")
# index.extend(doc)
#
# doc = Document("Oh! *crying* I'm so sorry. I'll try harder.")
# index.extend(doc)
#
# doc = Document("No skin off my teeth. Lets continue. The main descendent of lambda calculus is Lisp.")
# index.extend(doc)
#
# doc = Document("*weeping* lambda calculs goes into list...")
# index.extend(doc)
#
# doc = Document("Lisp! Not list!")
# index.extend(doc)
#
# doc = Document("Lisp!")
# index.extend(doc)
#
# doc = Document("Good. And most of the other programming languages descend from Turing machines.")
# index.extend(doc)
#

# doc = Document("Yes. I remember that. I think I'm getting the hang of it now.")
# index.extend(doc)
# doc = Document("Good. Lets continue. Remind me who invented lambda calculus?")
# index.extend(doc)
# doc = Document("Alonzo Church.")
# index.extend(doc)

# doc = Document("Very good! Now, what is the difference between lambda calculus and Turing machines?")
# index.extend(doc)
#
# doc = Document("Um, I don't know. Just the inventor? And the descendent languages?")
# index.extend(doc)

# doc = Document("Sure. Turing machines are finite state machines. Lambda calculus is a formal system for describing functions. Can you remember that?")
# index.extend(doc)

# doc = Document("Yeah I think so.")
# index.extend(doc)

# index.add_context("This is the global context")

# index = GPTMultiverseIndex.load_from_disk("test_index.json")
#
# index.clear_checkout()
# index.cherry_pick([17, 35])
#
# index.save_to_disk("test_index.json")
#
# print(index.index_struct.get_full_repr())
# print(index.short_path())
