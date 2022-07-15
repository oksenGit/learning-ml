class Human():
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def myfunc(self):
        print("Hello my name is " + self.name)


class Sport():
    def __init__(self, name, isOlympics):
        self.name = name
        self.isOlympics = isOlympics

# Create Athlete class that inherits from Human and Sport


class Athlete(Human, Sport):
    def __init__(self, name, age, sport, isOlympics):
        Human.__init__(self, name, age)
        Sport.__init__(self, sport, isOlympics)
        self.name = name
        self.sport = sport

    def myfunc(self):
        Human.myfunc(self)
        print("I'm an athlete")
        print("I'm a " + self.name)
        print("I'm " + str(self.age) + " years old")
        print("I'm a " + self.sport)
        if self.isOlympics:
            print("I'm an olympic athlete")


athlete = Athlete("John", 30, "swimming", True)
athlete.myfunc()
