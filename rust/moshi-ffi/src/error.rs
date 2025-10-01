use std::ffi::CString;
use std::os::raw::c_char;
use std::sync::Mutex;

/// Error codes returned by FFI functions
///
/// All fallible FFI functions return a `MoshiError` code. On error, a detailed message
/// is stored in thread-local storage and can be retrieved via `moshi_last_error`.
///
/// # Error Handling Pattern
///
/// ```c
/// MoshiError err = moshi_mimi_encode(...);
/// if (err != MoshiError_Ok) {
///     const char* msg = moshi_last_error();
///     if (msg) {
///         fprintf(stderr, "Error: %s\n", msg);
///         moshi_free_string((char*)msg);
///     }
///     return -1;
/// }
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoshiError {
    /// Operation completed successfully
    Ok = 0,
    /// A required pointer argument was null
    NullPointer = 1,
    /// An argument had an invalid value
    InvalidParameter = 2,
    /// Failed to load the model file
    ModelLoad = 3,
    /// Inference/model execution failed
    Inference = 4,
    /// Audio encoding failed
    Encode = 5,
    /// Token decoding failed
    Decode = 6,
    /// Memory allocation failed
    OutOfMemory = 7,
    /// An unknown error occurred (check `moshi_last_error` for details)
    Unknown = 99,
}

/// Global error storage for detailed error messages
static LAST_ERROR: Mutex<Option<String>> = Mutex::new(None);

/// Set the last error message
pub fn set_error(msg: String) {
    if let Ok(mut last_error) = LAST_ERROR.lock() {
        *last_error = Some(msg);
    }
}

/// Convert anyhow::Error to MoshiError and store the message
pub fn from_anyhow(err: anyhow::Error) -> MoshiError {
    let msg = format!("{:#}", err);
    set_error(msg);
    MoshiError::Unknown
}

/// Get the last error message as a C string
///
/// Retrieves the detailed error message for the most recent error. Error messages are stored
/// in a global mutex-protected storage that persists across calls until overwritten or cleared.
///
/// # Returns
/// - Non-null pointer to a null-terminated UTF-8 string containing the error message
/// - Null if no error has occurred or if the error message couldn't be converted to a C string
///
/// # Memory Ownership
/// The returned string is heap-allocated and **must** be freed by the caller using
/// `moshi_free_string`. The string remains valid until freed by the caller.
///
/// # Lifecycle
/// - Error messages persist until the next error occurs or `moshi_clear_error` is called
/// - Each call to this function allocates a new string, so call it once and store the result
/// - Multiple calls without an intervening error will return the same message text but
///   different heap allocations (all must be freed separately)
///
/// # Thread Safety
/// This function uses a global mutex for error storage. In multi-threaded applications,
/// error messages from different threads may overwrite each other. Retrieve and handle
/// errors immediately after the operation that caused them.
///
/// # Safety
/// The returned pointer must be freed with `moshi_free_string` to avoid memory leaks.
#[no_mangle]
pub unsafe extern "C" fn moshi_last_error() -> *const c_char {
    if let Ok(last_error) = LAST_ERROR.lock() {
        if let Some(ref msg) = *last_error {
            if let Ok(c_str) = CString::new(msg.as_str()) {
                return c_str.into_raw();
            }
        }
    }
    std::ptr::null()
}

/// Clear the last error message
///
/// Clears the stored error message. Subsequent calls to `moshi_last_error` will return null
/// until the next error occurs. This is useful for resetting error state between operations.
///
/// # Thread Safety
/// This function is thread-safe but affects global state. Clearing errors in one thread may
/// interfere with error handling in other threads.
#[no_mangle]
pub extern "C" fn moshi_clear_error() {
    if let Ok(mut last_error) = LAST_ERROR.lock() {
        *last_error = None;
    }
}

/// Free a C string returned by moshi functions
///
/// Frees a string that was allocated by `moshi_last_error`. After calling this function,
/// the pointer becomes invalid and must not be used.
///
/// # Parameters
/// - `ptr`: Pointer to a C string returned by `moshi_last_error`, or null
///
/// # Safety
/// - `ptr` must be null, or must have been returned by `moshi_last_error`
/// - `ptr` must not have been previously freed
/// - After this call, `ptr` becomes invalid and must not be dereferenced
///
/// # Thread Safety
/// This function is thread-safe. Different threads can free different strings concurrently.
#[no_mangle]
pub unsafe extern "C" fn moshi_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        let _ = CString::from_raw(ptr);
    }
}
