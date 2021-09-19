import streamlit as st


class MultiApp:
    """
    Framework for combining multiple streamlit applications.
    """

    def __init__(self):
        """
        Init function that holds apps.
        """
        self.apps = []

    def add_app(self, title, func):
        """
        Adds a new application.

        @param title: Title of the app. Appears in the dropdown in the sidebar.
        @param func: Yhe python function to render this app.
        """
        self.apps.append({
            "title": title,
            "function": func
        })

    def run(self):
        """
        Displays selected app.
        """
        app = st.sidebar.radio(
            'You can navigate at sidebar',
            self.apps,
            format_func=lambda app: app['title'])

        app['function']()
