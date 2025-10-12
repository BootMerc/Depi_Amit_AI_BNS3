from graphviz import Digraph

def create_animal_uml(output_path='animal_uml'):
    dot = Digraph(comment="Animal UML Diagram", format='png')
    dot.node("Animal", '''Animal
--------------------
+ describe()
+ make_sound() : abstract
''', shape="record")
    dot.node("Dog", '''Dog
--------------------
+ make_sound()
''', shape="record")
    dot.node("Cat", '''Cat
--------------------
+ make_sound()
''', shape="record")
    dot.node("Cow", '''Cow
--------------------
+ make_sound()
''', shape="record")
    dot.edge("Animal", "Dog")
    dot.edge("Animal", "Cat")
    dot.edge("Animal", "Cow")

    dot.render(output_path, cleanup=True)
    print(f"UML diagram saved as {output_path}.png")
