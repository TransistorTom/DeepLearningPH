class Monkey:
    def __init__(self, name, age):
        self.name = name  # Instance variable for monkey's name
        self.age = age    # Instance variable for monkey's age

    def introduce(self):
        print(f"Hi, I'm {self.name}, and I'm {self.age} years old!")

# Creating an instance of Monkey
gibbon = Monkey("Gibby", 5)

# Calling the method to introduce the monkey
gibbon.introduce()
