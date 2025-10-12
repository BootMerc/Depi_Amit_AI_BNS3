from dog import Dog
from cat import Cat
from cow import Cow
from diagram import create_animal_uml

if __name__ == "__main__":
    animals = [Dog(), Cat(), Cow()]

    for animal in animals:
        animal.describe()
        print(animal.make_sound())
        print("-" * 20)

    # Generate UML diagram
    create_animal_uml()
