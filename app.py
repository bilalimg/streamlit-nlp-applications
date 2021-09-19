from apps import home, LanguageIdentifier, SentimentAnalysis
from multiapp import MultiApp

app = MultiApp()

# Add all your application here
app.add_app("Home", home.app)
app.add_app("Language Identifier", LanguageIdentifier.app)
app.add_app("Sentiment Analysis", SentimentAnalysis.app)
# The main app
app.run()
