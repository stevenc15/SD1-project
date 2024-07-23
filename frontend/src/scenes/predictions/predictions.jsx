import { Canvas } from '@react-three/fiber'
import { useState } from 'react'
import Select from 'react-select'
import Model from '/public/Model'

function Predictions() {

  const [childData1, setChildData1] = useState([]);
  const [childData2, setChildData2] = useState([]);
  const [recStatus, setRecStatus] = useState("0")
  const [recStatusMessage, setRecStatMsg] = useState("Start Recording")
  const [frame1, setFrame1] = useState(0)
  const [frame2, setFrame2] = useState(0)
  const [curFrame, setCurrentFrame] = useState("")
  const [sensModel, setSensModel] = useState("6sensor")
  //console.log(childData1)
  //console.log(childData2)

  const options = [
    { value: '1sensor', label: '1 Sensor Model' },
    { value: '6sensor', label: '6 Sensor Model' }
  ]

  const jsonFileDownload = () => {
    const yourArray = childData1
    let json_data = JSON.stringify(yourArray);
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

  const jsonFileDownloadRec = () => {
    const yourArray2 = childData1.slice(frame1, frame2)
    console.log(yourArray2)
    let json_data = JSON.stringify(yourArray2);
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

  const updateRecStatus = () => {
    if(recStatus % 3 == 0){
      console.log("Recording started.")
      setRecStatus(1)
      setRecStatMsg("Stop Recording")
      setFrame1(curFrame)
      console.log(frame1)
    }
    if(recStatus % 3 == 1){
      console.log("Recording ended.")
      setRecStatus(2)
      setRecStatMsg("Download Recorded Data")
      setFrame2(curFrame)
      console.log(frame2)
    }
    if(recStatus % 3 == 2){
      console.log("Download recording.")
      setRecStatus(0)
      setRecStatMsg("Start Recording")
      jsonFileDownloadRec()
    }
  }

  return (
    <>
      <Select 
        defaultValue={sensModel}
        options={options}
        />
      <Canvas>
        <Model passChildData1={setChildData1} passChildData2={setChildData2} passCurrentFrame={setCurrentFrame}/>
      </Canvas>
      <button type="button">Rotate Left</button>
      <button type="button">Reset Camera</button>
      <button type="button">Rotate Right</button>
      <button type="button" onClick={updateRecStatus}>{recStatusMessage}</button>
      <button type="button" onClick={jsonFileDownload}>Download Full Recording</button>
    </>
  )
}

export default Predictions
