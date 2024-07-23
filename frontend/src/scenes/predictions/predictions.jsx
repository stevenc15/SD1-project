import { Canvas } from '@react-three/fiber'
import Model from '/public/Model'

function Predictions() {

  const jsonFileDownload = () => {
    const json_data = {
      name: "Dedar",
      age: "14",
      address: "House #28",
    };
    const fileName = "joint_data.json";
    const data = new Blob([JSON.stringify(json_data)], { type: "text/json" });
    const jsonURL = window.URL.createObjectURL(data);
    const link = document.createElement("a");
    document.body.appendChild(link);
    link.href = jsonURL;
    link.setAttribute("download", fileName);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <>
      <Canvas>
        <Model />
      </Canvas>
      <button type="button">Rotate Left</button>
      <button type="button">Reset Camera</button>
      <button type="button">Rotate Right</button>
      <button type="button">Start/Stop Recording</button>
      <button type="button" onClick={jsonFileDownload}>Download Recording</button>
    </>
  )
}

export default Predictions
