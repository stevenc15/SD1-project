<div id="ubuntu-gpu-docker-container">
  <h1>Ubuntu GPU Docker Container</h1>
  <p>This repository provides the necessary instructions and files to build and run a Docker container with Ubuntu and GPU support, including pre-cloning a Git repository.</p>

  <div id="prerequisites">
    <h2>Prerequisites</h2>
    <p>Ensure the following tools are installed on your system:</p>
    <ul>
      <li><strong>Docker</strong>: Follow the installation guide at <a href="https://docs.docker.com/get-docker/">Install Docker</a>.</li>
    </ul>
  </div>

  <div id="preparation">
    <h2>Preparation</h2>
    <p>Clone your desired Git repository into a local directory. This step ensures that your Docker container will have immediate access to the repository's contents.</p>
    <pre><code>git clone https://github.com/stevenc15/SD1-project.git
</code></pre>
  </div>

  <div id="build-the-docker-image">
    <h2>Build the Docker Image</h2>
    <p>Navigate to the directory containing your Dockerfile. Use the following command to build your Docker image:</p>
    <pre><code>docker build -t ubuntu-gpu:latest path/to/dockerfile/folder
</code></pre>
  </div>

  <div id="running-the-docker-container">
    <h2>Running the Docker Container</h2>
    <p>Run your Docker container with GPU support by executing:</p>
    <pre><code>docker run --gpus all -d -v $(pwd):/workspace ubuntu-gpu:latest
</code></pre>
    <p>This command mounts the current directory to <code>/workspace</code> inside the container, allowing you to work directly with the cloned repository files.</p>
  </div>

</div>
