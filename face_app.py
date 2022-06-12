from pages import home, inference, about
from multipage import MultiPage

app = MultiPage()


# add pages
app.add_page("home", home.app)
app.add_page("about", about.app)
app.add_page("use app", inference.app)

# run app
app.run()
