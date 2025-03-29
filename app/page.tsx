import Link from "next/link";
import Code from "./container/Code";

export default function Home() {
  return (
    <div>
      <h1 className="text-center font-bold text-6xl">
        Getting Started with Felix-detect-Fix
        &copy; Debayan Ghosh
      </h1>
        {/*
          model how to install the model how to check the validation how to try to generate the ipop program

          extension and CLI etc
        */}
      <p className="text-xl">
      <span className="text-2xl text-blue-400">What is felix-detect-fix?</span> All right so have you ever written a tons of code but got fluked with that one simple line no 12 error. I got you and really it's a huge problem to open 
      LLM to process your code and getting fixed. May be you are thinking about copilot, that is indeed good my tbh I dont have 10-20 USD to pay them. So here I code this huge project
      where anyone can use a product somewhere like copilot that can analyze your code and check that there is bug or not. if it has bug it can recommend you the patch. And according
      to our testing we have got more than 80% accuarcy.

      Now our second goal is building a model is good. but we know a huge amount of people they don't actually care about how the model works instead they want their job done easily. so
      I built an Vs code (visual studio code) extension for you where if you have a buggy code or there is a bug in your code that can be patched in place within a minute. and everything is 
      completely without internet.

      So I built this as a task project in my industrial project at INTEL. So in this following documentation I will guide you how you can use my project felix-detect-fix. So Let&apos; 
      get started.
      </p>
        <section>
          <h2 className="text-4xl mb-3 mt-2 font-semibold underline">Tutorial - 1 Use our model in your notebook to use:</h2>
          <div>
              <div className="text-3xl"><span className="text-4xl font-semibold underline mr-2">Requirements:</span> 
              <br/>
                <span className="text-4xl underline">STEP - 1: </span>
                If you want to run this on your local machine, make sure you have python installed in your machine. To check open terminal/powershell for windows
                or zsh/bash in macos or linux.

              <Code>
                python -v // for windows
              </Code>
              <Code>
                python3 -v // for macos or linux
              </Code>
            If you are getting something like this
            <Code>
              Python 3.12.3 (main, Feb  4 2025, 14:48:35) [GCC 13.3.0] on linux
            </Code>

                <span className="text-4xl underline">STEP - 2: </span>
                You are good to go now install the jupyter notebook [if you want to use VS code you can but its good to install jupyter notebook.]
                <Code>
                    pip install jupyterlab  
                </Code>
              <Code>
                    pip install notebook
              </Code>
              After Installing them try to create a notebook using this 
              <Code>
                jupyter notebook
              </Code>
              
                <span className="text-4xl underline">STEP - 3: </span>
                Now we will start writing some code, so at first we will import some packages like transformers, torch, textwrap (just for ease)
                They will help us to write less code.
                <Code>
                      {`
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
    import torch
    import textwrap
    `}                </Code>

                
                <span className="text-4xl underline">STEP - 4: </span>
                We will now download my <Link href="https://huggingface.co/felixoder/bug_detector_model" className="text-blue-500 underline">bug detector model</Link>

              <img src="/detector_hf.png" alt="detector" />
              you can manually download it but for you reference you can do it by-
                <Code>
              {`
    model_name = "felixoder/bug_detector_model"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

`}
                </Code>
            we're loading a pre-trained bug detection model (felixoder/bug_detector_model) using Hugging Face's Transformers library.
            The tokenizer processes raw code into tokens, while the model analyzes these tokens to classify whether the code contains bugs. 

                <span className="text-4xl underline">STEP - 5: </span>
            <Code>
              {`
    bug_fixer_model = "felixoder/bug_fixer_model"
    fixer_tokenizer = AutoTokenizer.from_pretrained(bug_fixer_model)
    fixer_model = AutoModelForCausalLM.from_pretrained(bug_fixer_model, torch_dtype=torch.float16, device_map="auto")
     
`}
            </Code>
                we are doing the same for the <Link href="https://huggingface.co/felixoder/bug_fixer_model" className="text-blue-500 underline">bug fixer model</Link>

              <img src="/fixer_hf.png" alt="fixer" />

                <span className="text-4xl underline">STEP - 6: </span>
              We will code a function that will just classify the code snippet with buggy or bug free ie, if the code contains bug then we will classify the code as buggy else we will
            classify the code as bug-free.
            <Code>
              {`
    
    def classify_code(code):
        """Classify input code as 'buggy' or 'bug-free' using the trained model."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)  # Move model to the correct device

        inputs = tokenizer(code, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).item()

        return "bug-free" if predicted_label == 0 else "buggy"



`}
            </Code>

                <span className="text-4xl underline">STEP - 7: </span>
            Now we will fix the code [if it is buggy] to a corrected version.
            <Code>
                {`
          def fix_buggy_code(code):
              """Generate a fixed version of the buggy code using the bug fixer model."""
              prompt = f"### Fix this buggy Python code:{code} just give the fixed code nothing else### Fixed Python code:"
              device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
              model.to(device)
              inputs = fixer_tokenizer(prompt, return_tensors="pt").to(device)

              with torch.no_grad():
                outputs = fixer_model.generate(
                  **inputs,
                  max_length=256,  # Reduce length for speed
                  do_sample=False,  # Make it deterministic
                  num_return_sequences=1  # Only one output
                )

              fixed_code = fixer_tokenizer.decode(outputs[0], skip_special_tokens=True)
              fixed_code = fixed_code.split("### Fixed Python code:")[1].strip() if "### Fixed Python code:" in fixed_code else fixed_code

              return textwrap.dedent(fixed_code).strip()
`}
</Code>

                <span className="text-4xl underline">STEP - 8: </span>
                Here we can test my model [how this works]
                so we will give input of a simple code snippet and test how it works.
            <Code>
              {`
    # Example buggy code input
    code_input = """
    for in nge(0, 9
      print(i)
    if val > 12:
      print("val {val} is greater")
    else:
      print("val {val} is less")
    """

    # Classify the code using the fine-tuned model
    status = classify_code(code_input)

    if status == "buggy":
      print("Buggy Code Detected")
      fixed_code = fix_buggy_code(code_input)
      print("Fixed Code:")
      print(fixed_code)
    else:
      print("Bug-free Code")
`}
            </Code>
            KABOOM!!! you have successfully tested my code now if you are a picky like me. Please sip your beer we have an accuracy of 80% you can see the evaluation 
            on my github <Link href="https://github.com/felixoder/bug_detection_ml_project" className="text-blue-500 underline">My project</Link>
            Now If your machine is getting hot like mine just do use Google colab or Kaggle Notebook. [Or rent a GPU from your friend].
                                    </div>

            <div className="lg:grid grid-cols-2 gap-2">
            </div>
          </div>
        </section>
      <section>
         <h2 className="text-4xl mb-3 mt-2 font-semibold underline">Tutorial - 2 Use My EXTENSION in your VS code WITHOUT INTERNET</h2>
          <div>
              <div className="text-3xl"><span className="text-4xl font-semibold underline mr-2">Requirements:</span> 
                You should have visual studio code installed in your computer if not install it from <Link href="https://code.visualstudio.com/download" className="text-blue-500 underline">Here</Link>
              <br/>
                <span className="text-4xl underline">STEP - 1: </span>
                  Open Your vs code and Press CTRL/CMD + N to create a new python file name it something like test.py and save the file somewhere in your computer.
                  Now go to your extension page (alternatively press CTRL/CMD + SHIFT + X) and search for <Link href="https://marketplace.visualstudio.com/items?itemName=DebayanGhosh.felix-detect-fix">felix-detect-fix</Link>
                Now install it in your machine. 
                <img src="/extension.png" alt="felix-detect-fix"/>
                
                <span className="text-4xl underline">STEP - 2: </span>
                After Installation for your reference open a file [the same folder where the test.py is ]
                and name it run_model.py [most important please dont give another name]
                and paste this follwing code.
                <Code>
              {`
    import os
    import sys
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )
    from huggingface_hub import snapshot_download  # Add this import

    # Get absolute paths relative to THIS file
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, "models")

    # Create model directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
  
    # Model configuration
    MODELS = {
        "detector": {
            "repo": "felixoder/bug_detector_model",
            "path": os.path.join(MODEL_DIR, "detector")
        },
        "fixer": {
            "repo": "felixoder/bug_fixer_model",
            "path": os.path.join(MODEL_DIR, "fixer")
        }
    }

    # Download models if missing
    for model in MODELS.values():
        if not os.path.exists(model["path"]):
            print(f"Downloading {model['repo']}...")
            snapshot_download(
                repo_id=model["repo"],
                local_dir=model["path"],
                local_dir_use_symlinks=False
            )

    # Now load the models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32

    # Load detector model
    detector_tokenizer = AutoTokenizer.from_pretrained(
        MODELS["detector"]["path"],
        local_files_only=True
    )
    detector_model = AutoModelForSequenceClassification.from_pretrained(
        MODELS["detector"]["path"],
        local_files_only=True,
        torch_dtype=torch_dtype
    ).to(device)

    # Load fixer model
    fixer_tokenizer = AutoTokenizer.from_pretrained(
        MODELS["fixer"]["path"],
        local_files_only=True
    )
    fixer_model = AutoModelForCausalLM.from_pretrained(
        MODELS["fixer"]["path"],
        local_files_only=True,
        torch_dtype=torch_dtype
    ).to(device)


    def classify_code(code):
        inputs = detector_tokenizer(
            code, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            outputs = detector_model(**inputs)
        predicted_label = torch.argmax(outputs.logits, dim=1).item()
        return "bug-free" if predicted_label == 0 else "buggy"



    def fix_buggy_code(code):
        prompt = f"### Fix this buggy Python code:\n{code}\n### Fixed Python code:\n"
        inputs = fixer_tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = fixer_model.generate(
                **inputs, max_length=256, do_sample=False, num_return_sequences=1
            )

        fixed_code = fixer_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return (
            fixed_code.split("### Fixed Python code:")[1].strip()
            if "### Fixed Python code:" in fixed_code
            else fixed_code
        )


    if __name__ == "__main__":
        command = sys.argv[1]
        code = sys.argv[2]

        if command == "classify":
            print(classify_code(code))
        elif command == "fix":
            print(fix_buggy_code(code))

`}
                </Code>

                <span className="text-4xl underline">STEP - 3: </span>
              After Installing this all datasets will be installed with the first run. So for running the code write some demo code on the test.py file. and then
            press CTRL/CMD+SHIFT+P and type <span className="text-blue-500">Detect Bug</span> or <span className="text-blue-500">Fix Bug</span> according to your choice. Now for the first time 
            it will take a bit time [10 -15 minutes  based on your machine speed and internet speed]
            and it will create the detector and fixer folder.
            <Code>
              {`
    models
      |
      |__detector
      |__fixer

    run_model.py
    test.py
`}
            </Code>
            And if your code has a bug you can see a pop up and the down right of your screen suggesting a <span className="text-orange-500">Fix This</span>
            You can click that one to fix your code <span className="text-blue-500">in-place.</span>
            
              </div>
          </div>


        If you still have Problem with the project please check this out
              </section>
      

    <footer className="border border-t-black flex flex-col justify-center items-center">
        If you have any question please check out (Give a star) 
        <a href="https://github.com/felixoder/bug_detection_ml_project" className="text-blue-500 underline">Bug Detector and fixer model</a>
        <a href="https://github.com/felixoder/felix-detect-fix" className="text-blue-500 underline">Visual Studio Extension</a>
        Made with love from Debayan &copy; 2025 
    </footer>
    </div>
  );
}
