# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def solution(a, b, c):
    for i in range(10, 101):
        if i % 3 == a and i % 5 == b and i % 7 == c:
            return i
    exit(1)


print(solution(2, 4, 5))
