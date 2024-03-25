// import multer from 'multer'
// import {GridFsStorage} from 'multer-gridfs-storage'

// Creating Storing Enginge
// Check the multer-gridfs-storage documentation for more.
// const storage = new GridFsStorage({
//     url: process.env.MONGO_STRING,
//     file: (req, file) => {
//       return new Promise((resolve, reject) => {
//         crypto.randomBytes(16, (err, buf) => {
//           if (err) {
//             return reject(err);
//           }
//           const filename = buf.toString('hex') + path.extname(file.originalname);
//           const fileInfo = {
//             filename: filename,
//             bucketName: 'uploads'
//           };
//           resolve(fileInfo);
//         });
//       });
//     }
//   });
//   const upload = multer({ storage });

//   export default upload