from apps import home, language_identifier, sentiment_analysis
from multiapp import MultiApp

app = MultiApp()

# Add all your application here
app.add_app("Home", home.app)
app.add_app("Language Identifier", language_identifier.app)
app.add_app("Sentiment Analysis", sentiment_analysis.app)
# The main app
app.run()
