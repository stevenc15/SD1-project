import React, { useEffect, useRef, useState } from "react";
import { useGLTF, useAnimations, Gltf } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import { Suspense } from "react";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { Controller, GUI } from "lil-gui";

export default function Model(props) {
  const group = useRef();
  const forearm = useRef();
  const { nodes, materials, animations } = useGLTF("/arm.gltf");
  const { actions } = useAnimations(animations, group);
  const modelPlaying = useRef(false);
  console.log(props.selectedModel);

  const [apiData, setData] = useState({
    timeData: [],
    angleData: [],
    dataLength: 0,
    timeElapsed: 0,
  });

  useEffect(() => {
    if (props.selectedModel === 1) {
      console.log("1 Sensor Output.");
      fetch("http://localhost:5000/joint_angle_pred", {
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
      })
        .then((res) => res.json())
        .then((data) => {
          console.log(data);

          const arrayColumn = (arr, n) => arr.map((x) => x[n]);
          let tmpLen = data.length;

          setData({
            timeData: arrayColumn(data, 0),
            angleData: arrayColumn(data, 1),
            dataLength: tmpLen,
            timeElapsed: data[tmpLen - 1][0] - data[0][0],
          });
        });
    } else if (props.selectedModel === 6) {
      console.log("6 Sensor Output.");
      fetch("http://localhost:5000/joint_angle_pred", {
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
      })
        .then((res) => res.json())
        .then((data) => {
          console.log(data);

          const arrayColumn = (arr, n) => arr.map((x) => x[n]);
          let tmpLen = data.length;

          setData({
            timeData: arrayColumn(data, 0),
            angleData: arrayColumn(data, 1),
            dataLength: tmpLen,
            timeElapsed: data[tmpLen - 1][0] - data[0][0],
          });
        });
    }
  }, []);

  console.log(apiData.timeElapsed + " of data loaded to Model.");

  const statObject = {
    xRot: 270,
    yRot: 180,
    zRot: 90,
    Framerate: 60,
    rotSpeed: 1,
    Interpolate: false,
  };

  let clock = new THREE.Clock();
  let delta = 0;
  let interval = 1 / statObject.Framerate;

  // console.log(group)
  // console.log(forearm)

  useEffect(() => {
    group.current.rotation.x = 270 * (Math.PI / 180);
    group.current.rotation.y = 180 * (Math.PI / 180);
    group.current.rotation.z = 90 * (Math.PI / 180);
    const gui = new GUI();
    gui.add(statObject, "xRot", 0, 360, 1);
    gui.add(statObject, "yRot", 0, 360, 1);
    gui.add(statObject, "zRot", 0, 360, 1);
    gui.add(statObject, "Framerate", 24, 120, 1);
    gui.add(statObject, "rotSpeed", 1, 5, 1);
    gui.add(statObject, "Interpolate");
    gui.onChange(() => {
      group.current.rotation.x = statObject.xRot * (Math.PI / 180);
      group.current.rotation.y = statObject.yRot * (Math.PI / 180);
      group.current.rotation.z = statObject.zRot * (Math.PI / 180);
      interval = 1 / statObject.Framerate;
      console.log(statObject.rotSpeed)
    });
    return () => {
      gui.destroy();
    };
  }, []);

  //onfinishchange

  useFrame(() => {
    delta += clock.getDelta();
    if (delta > interval) {
      if (props.isPlaying.current) {
        props.currentFrame.current = props.currentFrame.current + 1;
        if (props.currentFrame.current == apiData.dataLength - 1) {
          props.currentFrame.current = 0;
        }
      }
      forearm.current.skeleton.bones[2].rotation.x =
        apiData.angleData[props.currentFrame.current] * (Math.PI / 180);
      statObject.zRot =
        statObject.zRot + (props.rotDir.current * statObject.rotSpeed);
        console.log(statObject.rotSpeed)
      group.current.rotation.z = statObject.zRot * (Math.PI / 180);
    }
    delta = delta % interval;
  });

  useEffect(() => {
    props.passAngleData(apiData.angleData);
    props.passTimeData(apiData.timeData);
  }, [apiData]);

  return (
    <>
      <ambientLight />
      <OrbitControls enableZoom={true} />
      <Suspense fallback={null}>
        <group ref={group} {...props} dispose={null}>
          <group name="Sketchfab_Scene">
            <group name="Sketchfab_model" rotation={[-Math.PI / 2, 0, 0]}>
              <group name="root">
                <group name="GLTF_SceneRootNode" rotation={[Math.PI / 2, 0, 0]}>
                  <group
                    name="Armature_73"
                    position={[-0.123, -0.111, -1.771]}
                    rotation={[1.583, 0, 0]}
                  >
                    <group name="GLTF_created_0">
                      <primitive object={nodes.GLTF_created_0_rootJoint} />
                      <group name="Cube017_72" />
                      <skinnedMesh
                        ref={forearm}
                        name="Object_7"
                        geometry={nodes.Object_7.geometry}
                        material={materials.Cloth}
                        skeleton={nodes.Object_7.skeleton}
                      />
                    </group>
                  </group>
                </group>
              </group>
            </group>
          </group>
        </group>
      </Suspense>
    </>
  );
}

useGLTF.preload("/arm.gltf");
