// functions_js/index.js
//
// Creates a Firestore /users/{uid} doc when a new Firebase-Auth user is created
// ---------------------------------------------------------------------------

const functions = require("firebase-functions/v1"); // ⬅️  v1 API
const admin = require("firebase-admin");

admin.initializeApp();

exports.createUserProfile = functions.auth.user().onCreate(async (user) => {
  const db = admin.firestore();

  return db.collection("users").doc(user.uid).set({
    email: user.email || null,
    displayName: user.displayName || null,
    coin: 0, // default coin balance
    createdAt: admin.firestore.FieldValue.serverTimestamp(),
  });
});
