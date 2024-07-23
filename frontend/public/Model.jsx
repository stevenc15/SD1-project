import React, { useEffect, useRef, useState } from 'react'
import { useGLTF, useAnimations, Gltf } from '@react-three/drei'
import { useFrame } from '@react-three/fiber'
import { Suspense } from 'react'
import { OrbitControls } from '@react-three/drei'
import * as THREE from 'three'
import { GUI } from 'lil-gui'
//import data from '/public/joint_data'

export default function Model(props) {
  //const dataLen = data.numbers.length
  //console.log(dataLen)
  const group = useRef()
  const forearm = useRef()
  const { nodes, materials, animations } = useGLTF('/arm.gltf')
  const { actions } = useAnimations(animations, group)

  const [apiData, setData] = useState({
    values: [],
    dataLength: 0,
    timeElapsed: 0,
  })

  console.log(apiData)

  useEffect(() => {
    fetch("http://localhost:5000/joint_angle_pred", {
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      }
    })
      .then((res) => res.json())
      .then((data) => {
        console.log(data)
        console.log(data.length)
        console.log(data[0])
        console.log(data[3074])
        setData({
          // values: data.data_joint,
        })
      });
  }, []);

  // console.log(apiData.values)
  // let dataLength = (apiData.values).length
  // console.log(dataLength)

  const statObject = {
    xRot: 0,
    yRot: 0,
    zRot: 0,
    armAng: 0,
    Framerate: 60,
    Interpolate: false,
    PausePlay: true,
  }

  let clock = new THREE.Clock()
  let delta = 0
  let interval = 1 / statObject.Framerate
  let currentFrame = 0

  //console.log(group)
  //console.log(forearm)

  useEffect(() => {
    const gui = new GUI()
    gui.add(statObject, 'xRot', 0, 360, 2)
    gui.add(statObject, 'yRot', 0, 360, 2)
    gui.add(statObject, 'zRot', 0, 360, 2)
    //gui.add(forearm.current.skeleton.bones[2].rotation, 'x', Math.PI *(3/2), Math.PI * 2)
    gui.add(statObject, 'armAng', 0, 180)
    gui.add(statObject, 'Framerate', 24, 120, 1)
    gui.add(statObject, 'Interpolate')
    gui.add(statObject, 'PausePlay')
    gui.onChange(event => {
      group.current.rotation.x = statObject.xRot * (Math.PI / 180)
      group.current.rotation.y = statObject.yRot * (Math.PI / 180)
      group.current.rotation.z = statObject.zRot * (Math.PI / 180)
      forearm.current.skeleton.bones[2].rotation.x = (statObject.armAng - 90) * (Math.PI / 180)
      interval = 1 / statObject.Framerate
      console.log(apiData.values)
    })
    return () => {
      gui.destroy()
    }
  }, [])

  //onfinishchange

  useFrame(() => {
    if (statObject.Interpolate == false) {
      delta += clock.getDelta();
      if (delta > interval) {
        if (statObject.PausePlay == true)
          {
            currentFrame = currentFrame + 1
          }
        //console.log(currentFrame)
        //console.log(statObject.PausePlay)
        //forearm.current.skeleton.bones[2].rotation.x = (apiData.values[currentFrame]) * (Math.PI / 180)
      }
      delta = delta % interval
      // if (currentFrame > dataLength) {
      //   currentFrame = 0
      // }
    }
  })

  // function rotLeft()
  // {
  //   group.rotate =- 90
  // }

  // function rotRight()
  // {
  //   group.rotate =+ 90
  // }

  // function resetCam()
  // {
  //   group.rotate = 0
  // }

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
                  <group name="Armature_73" position={[-0.123, -0.111, -1.771]} rotation={[1.583, 0, 0]}>
                    <group name="GLTF_created_0">
                      <primitive object={nodes.GLTF_created_0_rootJoint} />
                      <group name="Cube017_72" />
                      <skinnedMesh ref={forearm} name="Object_7" geometry={nodes.Object_7.geometry} material={materials.Cloth} skeleton={nodes.Object_7.skeleton} />
                    </group>
                  </group>
                </group>
              </group>
            </group>
          </group>
        </group>
      </Suspense>
    </>
  )
}

useGLTF.preload('/arm.gltf')