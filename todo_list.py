def display_menu():
    print("\nTo-Do List Options:")
    print("1. Add a new task")
    print("2. View all tasks")
    print("3. Mark a task as completed")
    print("4. Delete a task")
    print("5. Exit")

def add_task(tasks):
    task_name = input("Enter the new task name: ")
    if task_name in tasks:
        print("Task already exists.")
    else:
        tasks[task_name] = False
        print("Task added.")

def view_tasks(tasks):
    if tasks:
        print("\nYour To-Do List:")
        for task, completed in tasks.items():
            status = "Completed" if completed else "Not Completed"
            print(f"{task}: {status}")
    else:
        print("Your To-Do List is empty.")

def mark_task_completed(tasks):
    task_name = input("Enter the task name to mark as completed: ")
    if task_name in tasks:
        tasks[task_name] = True
        print("Task marked as completed.")
    else:
        print("Task not found.")

def delete_task(tasks):
    task_name = input("Enter the task name to delete: ")
    if task_name in tasks:
        del tasks[task_name]
        print("Task deleted.")
    else:
        print("Task not found.")

def main():
    tasks = {}

    while True:
        display_menu()
        choice = input("Choose an option: ")

        if choice == '1':
            add_task(tasks)
        elif choice == '2':
            view_tasks(tasks)
        elif choice == '3':
            mark_task_completed(tasks)
        elif choice == '4':
            delete_task(tasks)
        elif choice == '5':
            print("Exiting To-Do List...")
            break
        else:
            print("Invalid choice. Please choose again.")

if __name__ == "__main__":
    main()
