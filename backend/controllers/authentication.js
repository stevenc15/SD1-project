import User from "../models/user-model.js";
import bcryptjs from "bcryptjs"

export const signup = async (req, res, next) => {
    const {userName, email, password } = req.body
    const hashedPassword = bcryptjs.hashSync(password, 10)
    const newUser = new User({userName, email, password: hashedPassword})
    try {
        await newUser.save()
        res.status(201).json("User created succesfully")
    } catch (error) {
        next(error)
        console.log("Something went wrong")
    }
}

export const login = async (req, res, next) => {
    const {userName, password} = req.body
    try {
        const validUser = await User.findOne({userName})
        if (!validUser)
            return res.status(400).json("Invalid user name")
        const validPassword = await bcryptjs.compare(password, validUser.password)
        if (!validPassword)
            return res.status(400).json("Invalid Password!")
        res.status(200).json("Logged in succesfully")
    } catch (error) {
        next(error)
        console.log("Something went wrong")
    }
}