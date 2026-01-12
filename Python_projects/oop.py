class Employee:
    def __init__(self, name, age, role, marks):
        self.name = name
        self.age = age
        self.role = role
        self.marks = marks
    @staticmethod
    def hello():
        print("Hello")
    def avg_mark(self):
        print("Hi! "+self.name+ "\nYour age is ", self.age, "\nYou are a brilliant "+self.role+ "\nYour avg score is :", (sum(self.marks))/len(self.marks))
e1 = Employee("Randy", 31, "Ml Engineer", [23, 30 ,61])

e1.avg_mark()
e1.hello()