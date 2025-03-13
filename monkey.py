class Monkey:
    def __init__(self, name, age, tree):
        self.name = name          # Instance variable for monkey's name
        self.age = age            # Instance variable for monkey's age
        self.favourite_tree= tree # Instance variable for monkey's favourite tree
        
    def introduce(self):
        print(f"Hi, I'm {self.name}, I'm {self.age} years old, and I love {self.favorite_tree} trees!")
        
    def celebrate_birthday(self):
        self.age += 1  # Modify THIS monkey's age
        print(f"Happy birthday, {self.name}! You are now {self.age} years old!")

    def swing(self):
        print(f"{self.name} is swinging through the trees!")

    
# Creating an instance of Monkey (Feel free to add your own monkey to the jungle! give it a "name", age, "favourite_tree")
gibbon = Monkey("Gibby", 5, "banyan")
lemur = Monkey("Momo", 3, "fig")
panda = Monkey("Xi Lan", 12, "bamboo")
