import sys
import subprocess

def main():
    #run_script(script_name):
    
    scripts = ["trainer.py", "point_cloud.py",'camera.py']

    print('press 1 for trainer,2 for point cloud, 3 for camera')

    for i, script in enumerate(scripts):
        print(f"{i + 1}: {script}")

    key = input()
    while key != 'q':
        try:
            index = int(key) - 1
            if index < 0 or index >= len(scripts):
                raise ValueError
        except ValueError:
            print("Invalid input")
            sys.exit()
        script_name = scripts[index]
        subprocess.run(["python", script_name])
        key = input('press 1 for trainer,2 for point cloud, 3 for camera or q for quit')
if __name__ == "__main__":
    main()


